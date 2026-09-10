"""ACF/PACF, otomatik ARIMA/SARIMA secimi, artik tanilari ve ongoru."""
from __future__ import annotations

import time
import warnings

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf

from panel.utils import f, star
from .prepare import TSData

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# Korelogramlar
# --------------------------------------------------------------------------

def correlogram(y, nlags=None):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 8:
        return None
    if nlags is None:
        nlags = int(min(max(10, n // 4), 40, n - 2))
    try:
        a, aci, qstat, qp = acf(y, nlags=nlags, alpha=0.05, qstat=True, fft=True)
        p = pacf(y, nlags=min(nlags, n // 2 - 1), alpha=None, method="ywadjusted")
    except Exception:
        return None
    band = 1.959963985 / np.sqrt(n)
    rows = []
    for k in range(1, min(len(a), len(p)) ):
        rows.append({
            "lag": k, "acf": f(a[k], 4), "pacf": f(p[k], 4),
            "q": f(qstat[k - 1]) if k - 1 < len(qstat) else None,
            "qP": f(qp[k - 1]) if k - 1 < len(qp) else None,
            "sigAcf": bool(abs(a[k]) > band),
            "sigPacf": bool(abs(p[k]) > band),
        })
    return {"n": int(n), "band": f(band, 4), "rows": rows,
            "nlags": len(rows)}


# --------------------------------------------------------------------------
# Otomatik model secimi
# --------------------------------------------------------------------------

def _aicc(res, k, n):
    try:
        aic = float(res.aic)
    except Exception:
        return np.inf
    den = n - k - 1
    return aic + (2 * k * (k + 1) / den) if den > 0 else aic


def _fit(y, order, seasonal, trend, maxiter=100):
    return SARIMAX(y, order=order, seasonal_order=seasonal, trend=trend,
                   enforce_stationarity=False, enforce_invertibility=False
                   ).fit(disp=False, maxiter=maxiter)


def ndiffs(y, max_d=2, alpha=0.05):
    """ADF/KPSS ile gereken fark alma derecesini belirler."""
    from statsmodels.tsa.stattools import adfuller, kpss
    x = np.asarray(y, float)
    x = x[np.isfinite(x)]
    for d in range(0, max_d + 1):
        if x.size < 10:
            return d
        adf_ns = True     # ADF: H0 birim kok -> reddedilemezse duragan degil
        kp_ns = False     # KPSS: H0 duraganlik -> reddedilirse duragan degil
        try:
            adf_ns = bool(adfuller(x, autolag="AIC")[1] > alpha)
        except Exception:
            pass
        try:
            kp_ns = bool(kpss(x, regression="c", nlags="auto")[1] < alpha)
        except Exception:
            pass
        if not (adf_ns or kp_ns):
            return d
        x = np.diff(x)
    return max_d


def nsdiffs(y, period, threshold=0.64):
    """
    Mevsimsel fark derecesi (Hyndman & Athanasopoulos mevsimsellik gucu olcutu).
    Fs = max(0, 1 - Var(kalinti) / Var(mevsimsel + kalinti)); Fs > 0.64 ise D = 1.
    """
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if period < 2 or y.size < 2 * period + 4:
        return 0, None
    try:
        from statsmodels.tsa.seasonal import STL
        r = STL(y, period=int(period), robust=True).fit()
        rem = np.asarray(r.resid, float)
        sea = np.asarray(r.seasonal, float)
        den = float(np.var(sea + rem))
        if den <= 0:
            return 0, 0.0
        fs = max(0.0, 1.0 - float(np.var(rem)) / den)
        return (1 if fs > threshold else 0), float(fs)
    except Exception:
        return 0, None


def auto_arima(y, d=None, max_p=3, max_q=3, period=1, seasonal=True,
               max_P=1, max_Q=1, ic="aicc", trend_const=True, time_budget=None):
    """
    Izgara aramasi ile en iyi ARIMA/SARIMA modelini secer.

    ONEMLI: Farkli fark alma dereceleri (d, D) ile tahmin edilen modellerin
    bilgi olcutleri karsilastirilamaz; cunku modeller farkli donusturulmus
    serilere uyarlanir. Bu nedenle d birim kok testleriyle, D ise mevsimsellik
    gucu olcutuyle ONCE sabitlenir; izgara aramasi yalnizca (p, q, P, Q)
    uzerinde yapilir.
    """
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 15:
        return None, [], {}

    dd = int(d) if d is not None else ndiffs(y)
    can_season = bool(seasonal and period > 1 and n >= 3 * period + 10)
    if can_season:
        D, seas_strength = nsdiffs(y, period)
    else:
        D, seas_strength = 0, None

    # Mevsimsellik zayifsa mevsimsel terimler hic denenmez. Bu, izgarayi
    # (max_P+1)*(max_Q+1) kat kucultur: ceyreklik veride 64 tahminden 16'ya
    # duser. Ucretsiz sunucu kaynaklarinda sure farki belirleyicidir.
    weak_season = bool(can_season and D == 0 and
                       (seas_strength is None or seas_strength < 0.30))
    use_seasonal = bool(can_season and not weak_season)

    # asiri fark almayi engelle
    if dd + D > 2:
        dd = max(0, 2 - D)

    P_range = range(0, max_P + 1) if use_seasonal else [0]
    Q_range = range(0, max_Q + 1) if use_seasonal else [0]
    trend = "c" if (trend_const and dd <= 1 and D == 0) else "n"

    # Bellek: 64 SARIMAXResults nesnesini ayni anda tutmak yuzlerce MB'a
    # ulasabilir. Bu yuzden arama sirasinda yalnizca skorlar saklanir,
    # kazanan model en sonda bir kez yeniden tahmin edilir.
    # Izgara, toplam parametre sayisina gore sadeden karmasiga siralanir.
    # Sure butcesi dolarsa arama kesilir; bu durumda geride birakilanlar en
    # karmasik modeller olur, dolayisiyla kesinti sonucun kalitesini en az
    # etkileyecek noktadan yapilmis olur.
    grid = []
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            for P in P_range:
                for Q in Q_range:
                    if p + q + P + Q == 0 and dd == 0 and D == 0:
                        continue
                    grid.append((p + q + 2 * (P + Q), p, q, P, Q))
    grid.sort()

    # Zaman butcesi. Butcenin bir kismi kazananin yeniden tahmini icin ayrilir;
    # ayrica bir sonraki tahminin butceyi asacagi ongoruluyorsa arama hic
    # baslatilmaz. Boylece toplam sure yavas sunucularda da butceyi asmaz —
    # tek asilan istek tarayicida "Failed to fetch" olarak gorunuyordu.
    budget = float(time_budget) if time_budget else None
    t_start = time.time()
    search_end = (t_start + budget * 0.75) if budget else None
    slowest = 0.0
    cands = []
    truncated = False
    for _, p, q, P, Q in grid:
        if search_end:
            now = time.time()
            if now + slowest * 0.8 > search_end:
                truncated = now > search_end or bool(cands)
                if truncated:
                    break
        order = (p, dd, q)
        seas = (P, D, Q, period) if use_seasonal else (0, 0, 0, 0)
        t_fit = time.time()
        try:
            res = _fit(y, order, seas, trend)
            k = int(len(res.params))
            score = _aicc(res, k, n) if ic == "aicc" else \
                (float(res.bic) if ic == "bic" else float(res.aic))
            aic, bic = float(res.aic), float(res.bic)
        except Exception:
            continue
        finally:
            res = None              # sonuc nesnesini hemen serbest birak
            slowest = max(slowest, time.time() - t_fit)
        if not np.isfinite(score):
            continue
        cands.append({"order": order, "seasonal": seas, "trend": trend,
                      "score": score, "aic": aic, "bic": bic})

    if not cands:
        return None, [], {}
    cands.sort(key=lambda c: c["score"])

    # kazananı yeniden tahmin et (yalnızca bir sonuç nesnesi bellekte kalır)
    best = cands[0]
    # kalan sureye gore yakinsama adimi: butce daralmissa daha az yineleme
    left = (t_start + budget) - time.time() if budget else None
    refit_iter = 300 if (left is None or left > 3 * max(slowest, 0.05)) else 150
    try:
        best["res"] = _fit(y, best["order"], best["seasonal"], best["trend"],
                           maxiter=refit_iter)
    except Exception:
        return None, [], {}

    meta = {"d": int(dd), "D": int(D), "period": int(period),
            "seasonalUsed": bool(use_seasonal),
            "seasonalSkipped": bool(weak_season),
            "seasonalStrength": f(seas_strength, 3) if seas_strength is not None else None,
            "dSource": "kullanıcı" if d is not None else "ADF/KPSS testleri",
            "nFitted": len(cands), "nGrid": len(grid), "truncated": truncated,
            "searchSeconds": round(time.time() - t_start, 2)}
    return best, cands[:6], meta


def _order_str(order, seasonal):
    s = f"ARIMA({order[0]},{order[1]},{order[2]})"
    if seasonal and seasonal[3] > 1 and any(seasonal[:3]):
        s = (f"SARIMA({order[0]},{order[1]},{order[2]})"
             f"({seasonal[0]},{seasonal[1]},{seasonal[2]})[{seasonal[3]}]")
    return s


def burn_in(res, order=None, seasonal=None):
    """
    Durum-uzayi suzgecinin dagilimli (diffuse) baslangic gozlemlerinin sayisi.

    SARIMAX'te ilk d + D*s (+ p) gozlem icin uretilen artiklar suzgec heniz
    yakinsamadigi icin cok buyuktur. Bu gozlemler uyum olcutlerine ve artik
    tanilarina dahil edilirse RMSE sisirilir, Ljung-Box testi yanlislikla
    reddedilir. Bu nedenle hem tanilarda hem de uyum olcutlerinde atlanir.
    """
    k = 0
    for attr in ("loglikelihood_burn", "nobs_diffuse"):
        v = getattr(res, attr, None)
        try:
            if v is not None and int(v) > k:
                k = int(v)
        except (TypeError, ValueError):
            pass
    if order is not None:
        d = int(order[1])
        s = int(seasonal[3]) if seasonal and len(seasonal) > 3 else 0
        D = int(seasonal[1]) if seasonal else 0
        k = max(k, d + D * s + int(order[0]))
    return int(k)


def diagnostics(res, period=1, burn=0):
    e = np.asarray(res.resid, float)
    if burn > 0 and e.size > burn + 8:
        e = e[burn:]
    e = e[np.isfinite(e)]
    n = e.size
    out = {}
    if n < 10:
        return out

    lag = int(min(max(10, 2 * period), max(5, n // 5)))
    try:
        lb = acorr_ljungbox(e, lags=[lag], return_df=True)
        out["ljungBox"] = {
            "name": f"Ljung-Box Q({lag})", "lag": lag,
            "stat": f(float(lb["lb_stat"].iloc[0])),
            "p": f(float(lb["lb_pvalue"].iloc[0])),
            "h0": "Artıklarda otokorelasyon yoktur",
            "ok": bool(float(lb["lb_pvalue"].iloc[0]) > 0.05)}
    except Exception:
        pass
    try:
        jb, jbp = stats.jarque_bera(e)[:2]
        out["jarqueBera"] = {"name": "Jarque-Bera", "stat": f(jb), "p": f(jbp),
                             "h0": "Artıklar normal dağılmaktadır",
                             "ok": bool(jbp > 0.05)}
    except Exception:
        pass
    try:
        a = het_arch(e, nlags=min(lag, max(1, n // 5)))
        out["archLM"] = {"name": f"ARCH-LM({min(lag, max(1, n // 5))})",
                         "stat": f(a[0]), "p": f(a[1]),
                         "h0": "Artıklarda ARCH etkisi (değişen varyans) yoktur",
                         "ok": bool(a[1] > 0.05)}
    except Exception:
        pass
    dw = float(((np.diff(e) ** 2).sum()) / (e @ e)) if (e @ e) > 0 else np.nan
    out["durbinWatson"] = {"name": "Durbin-Watson", "stat": f(dw),
                           "ok": bool(1.5 <= dw <= 2.5)}
    ok = [v.get("ok") for v in out.values() if "ok" in v]
    out["allOk"] = bool(ok and all(ok))
    return out


def run(ts: TSData, var, d=None, max_p=3, max_q=3, seasonal=True,
        horizon=5, ic="aicc", nlags=None, time_budget=45):
    y = ts.y(var)
    n = y.size
    if n < 15:
        return None

    best, top, meta = auto_arima(y, d=d, max_p=max_p, max_q=max_q,
                                 period=ts.period, seasonal=seasonal, ic=ic,
                                 time_budget=time_budget)
    if not best:
        return None
    res = best["res"]
    order, seas = best["order"], best["seasonal"]

    names = list(res.param_names)
    params = np.asarray(res.params, float)
    bse = np.asarray(res.bse, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(bse > 0, params / bse, np.nan)
    pv = 2 * stats.norm.sf(np.abs(z))
    coeffs = [{"name": names[i], "B": f(params[i]), "SE": f(bse[i]),
               "z": f(z[i]), "p": f(pv[i]), "sig": star(pv[i])}
              for i in range(len(names))]

    fitted = np.asarray(res.fittedvalues, float)
    resid = np.asarray(res.resid, float)
    burn = burn_in(res, order, seas)
    if burn >= y.size - 5:
        burn = 0
    m = np.isfinite(y) & np.isfinite(fitted)
    if burn > 0:
        m[:burn] = False          # diffuse başlangıç gözlemlerini hariç tut
    rmse = float(np.sqrt(np.mean((y[m] - fitted[m]) ** 2))) if m.any() else np.nan
    mae = float(np.mean(np.abs(y[m] - fitted[m]))) if m.any() else np.nan
    nz = m & (np.abs(y) > 1e-12)
    mape = float(np.mean(np.abs((y[nz] - fitted[nz]) / y[nz])) * 100) if nz.any() else None

    fc = None
    if horizon and horizon > 0:
        try:
            fo = res.get_forecast(steps=int(horizon))
            mean = np.asarray(fo.predicted_mean, float)
            ci = np.asarray(fo.conf_int(alpha=0.05), float)
            step = _next_labels(ts, int(horizon))
            fc = {"horizon": int(horizon), "labels": step,
                  "mean": [f(x, 6) for x in mean],
                  "lower": [f(x, 6) for x in ci[:, 0]],
                  "upper": [f(x, 6) for x in ci[:, 1]]}
        except Exception:
            fc = None

    return {
        "variable": var,
        "model": _order_str(order, seas),
        "order": list(order), "seasonalOrder": list(seas), "trend": best["trend"],
        "coeffs": coeffs,
        "fit": {"aic": f(res.aic), "bic": f(res.bic),
                "hqic": f(getattr(res, "hqic", np.nan)),
                "logLik": f(res.llf), "sigma2": f(getattr(res, "mse", np.nan)),
                "rmse": f(rmse), "mae": f(mae), "mape": f(mape),
                "nobs": int(res.nobs)},
        "candidates": [{"model": _order_str(c["order"], c["seasonal"]),
                        "aic": f(c["aic"]), "bic": f(c["bic"]),
                        "score": f(c["score"])} for c in top],
        "selection": meta,
        "burnIn": int(burn),
        "diagnostics": diagnostics(res, ts.period, burn),
        "correlogram": correlogram(y, nlags),
        "residCorrelogram": correlogram(resid[burn:] if burn else resid, nlags),
        "series": {"labels": ts.labels, "actual": [f(x, 6) for x in y],
                   "fitted": [None if (i < burn or not np.isfinite(x)) else f(x, 6)
                              for i, x in enumerate(fitted)],
                   "resid": [None if (i < burn or not np.isfinite(x)) else f(x, 6)
                             for i, x in enumerate(resid)]},
        "forecast": fc,
        "ic": ic,
    }


def _next_labels(ts: TSData, h):
    """Ongoru donemleri icin etiket uretir."""
    last = ts.labels[-1] if ts.labels else ""
    out = []
    try:
        if ts.freq == "A":
            base = int(float(last))
            return [str(base + i) for i in range(1, h + 1)]
        if ts.freq == "Q" and "Q" in str(last).upper():
            yr, q = str(last).upper().split("Q")
            yr, q = int(yr), int(q)
            for _ in range(h):
                q += 1
                if q > 4:
                    q = 1; yr += 1
                out.append(f"{yr}Q{q}")
            return out
        if ts.freq == "M" and "-" in str(last):
            yr, mo = str(last).split("-")[:2]
            yr, mo = int(yr), int(mo)
            for _ in range(h):
                mo += 1
                if mo > 12:
                    mo = 1; yr += 1
                out.append(f"{yr}-{mo:02d}")
            return out
    except Exception:
        pass
    return [f"t+{i}" for i in range(1, h + 1)]
