"""Panel birim kok testleri: LLC, IPS, Fisher-ADF, Fisher-PP, Hadri."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import stats

from .prepare import Panel
from .utils import (adf_regression, bic_lag_select, f, mackinnon_adf_pvalue,
                    newey_west_lrv, phillips_perron)

TREND_LABEL = {"c": "Sabitli", "ct": "Sabitli ve Trendli", "n": "Sabitsiz"}


# --------------------------------------------------------------------------
# Hizli ADF cekirdegi (Monte Carlo icin)
# --------------------------------------------------------------------------

def _design(y, p, trend):
    dy = np.diff(y)
    m = dy.size
    if m - p < 4:
        return None, None, 0
    Y = dy[p:]
    n = Y.size
    cols = []
    ndet = 0
    if trend in ("c", "ct"):
        cols.append(np.ones(n)); ndet += 1
    if trend == "ct":
        cols.append(np.arange(1, n + 1, dtype=float)); ndet += 1
    cols.append(y[p:-1])
    for j in range(1, p + 1):
        cols.append(dy[p - j: m - j])
    return Y, np.column_stack(cols), ndet


def _fast_adf_t(y, p, trend):
    Y, Z, ndet = _design(np.asarray(y, float), p, trend)
    if Y is None or Z.shape[0] <= Z.shape[1] + 1:
        return np.nan
    try:
        G = Z.T @ Z
        Gi = np.linalg.inv(G)
        b = Gi @ (Z.T @ Y)
    except np.linalg.LinAlgError:
        return np.nan
    r = Y - Z @ b
    dof = Z.shape[0] - Z.shape[1]
    s2 = float(r @ r) / dof
    v = s2 * Gi[ndet, ndet]
    if v <= 0:
        return np.nan
    return float(b[ndet] / np.sqrt(v))


@lru_cache(maxsize=256)
def _ips_moments(T: int, p: int, trend: str, reps: int = 2000, seed: int = 20240501):
    """IPS t-bar istatistigi icin bireysel ADF t-degerlerinin null momentleri."""
    rng = np.random.default_rng(seed + 1000 * T + 10 * p + len(trend))
    vals = np.empty(reps)
    for r in range(reps):
        y = np.cumsum(rng.standard_normal(T))
        vals[r] = _fast_adf_t(y, p, trend)
    v = vals[np.isfinite(vals)]
    if v.size < 30:
        return (-1.5, 1.0)
    return (float(v.mean()), float(v.var(ddof=1)))


# --------------------------------------------------------------------------
# Levin-Lin-Chu
# --------------------------------------------------------------------------

def _llc_pooled_t(series, p, trend):
    """LLC havuzlanmis t istatistigi (ortogonallestirilmis ve olceklenmis)."""
    E, V = [], []
    for y in series:
        y = np.asarray(y, float)
        dy = np.diff(y)
        m = dy.size
        if m - p < 5:
            continue
        Y = dy[p:]
        n = Y.size
        ylag = y[p:-1]
        aux = []
        if trend in ("c", "ct"):
            aux.append(np.ones(n))
        if trend == "ct":
            aux.append(np.arange(1, n + 1, dtype=float))
        for j in range(1, p + 1):
            aux.append(dy[p - j: m - j])
        if aux:
            W = np.column_stack(aux)
            try:
                Wi = np.linalg.pinv(W.T @ W)
            except np.linalg.LinAlgError:
                continue
            P = W @ (Wi @ W.T)
            e_hat = Y - P @ Y
            v_hat = ylag - P @ ylag
        else:
            e_hat, v_hat = Y, ylag
        k = (W.shape[1] + 1) if aux else 1
        dof = n - k
        if dof < 3:
            continue
        # bireysel ADF regresyonundan standart hata
        t_i, rho_i, se_i, resid, ni, ki = adf_regression(y, p, trend)
        if resid is None or ni <= ki:
            continue
        sig = float(np.sqrt((resid @ resid) / (ni - ki)))
        if sig <= 0:
            continue
        E.append(e_hat / sig)
        V.append(v_hat / sig)
    if len(E) < 2:
        return np.nan
    e = np.concatenate(E)
    v = np.concatenate(V)
    den = float(v @ v)
    if den <= 0:
        return np.nan
    delta = float(v @ e) / den
    r = e - delta * v
    dof = e.size - 1
    s2 = float(r @ r) / dof
    se = np.sqrt(s2 / den)
    if se <= 0:
        return np.nan
    return float(delta / se)


@lru_cache(maxsize=128)
def _llc_null(N: int, T: int, p: int, trend: str, reps: int = 300,
              seed: int = 771103):
    rng = np.random.default_rng(seed + 977 * N + 31 * T + 7 * p + len(trend))
    out = np.empty(reps)
    for r in range(reps):
        series = [np.cumsum(rng.standard_normal(T)) for _ in range(N)]
        out[r] = _llc_pooled_t(series, p, trend)
    v = out[np.isfinite(out)]
    return v


def llc_test(series, p, trend, reps=300):
    stat = _llc_pooled_t([np.asarray(s, float) for s in series], p, trend)
    if not np.isfinite(stat):
        return None
    Ts = [len(s) for s in series]
    T = int(np.median(Ts))
    N = len(series)
    null = _llc_null(N, T, p, trend, int(reps))
    if null.size < 30:
        return {"test": "Levin-Lin-Chu", "stat": f(stat), "p": None,
                "dist": "—", "note": "Null dağılım simüle edilemedi."}
    pval = float((null <= stat).mean())
    z = (stat - float(null.mean())) / float(null.std(ddof=1))
    return {"test": "Levin-Lin-Chu (LLC)",
            "stat": f(stat), "z": f(z), "p": f(pval),
            "dist": "Monte Carlo (N(0,1)'e standartlaştırılmış)",
            "reps": int(null.size),
            "h0": "Ortak birim kök vardır (δ = 0)",
            "sig": bool(pval < 0.05)}


# --------------------------------------------------------------------------
# Im-Pesaran-Shin
# --------------------------------------------------------------------------

def ips_test(series, p, trend, reps=1500):
    ts, Es, Vs, used = [], [], [], 0
    for s in series:
        y = np.asarray(s, float)
        T = y.size
        t = _fast_adf_t(y, p, trend)
        if not np.isfinite(t):
            continue
        mu, var = _ips_moments(int(T), int(p), trend, int(reps))
        ts.append(t); Es.append(mu); Vs.append(var); used += 1
    if used < 2:
        return None
    ts = np.asarray(ts)
    tbar = float(ts.mean())
    num = float(ts.sum() - np.sum(Es))
    den = float(np.sqrt(np.sum(Vs)))
    if den <= 0:
        return None
    W = num / den
    pval = float(stats.norm.cdf(W))
    return {"test": "Im-Pesaran-Shin (IPS) W-t̄",
            "stat": f(W), "tbar": f(tbar), "p": f(pval),
            "dist": "N(0,1)", "N": used,
            "h0": "Tüm birimler için birim kök vardır",
            "sig": bool(pval < 0.05)}


# --------------------------------------------------------------------------
# Fisher tipi testler
# --------------------------------------------------------------------------

def _fisher(pvals, label, h0):
    pv = np.asarray([q for q in pvals if q is not None and np.isfinite(q)], float)
    pv = np.clip(pv, 1e-12, 1 - 1e-12)
    N = pv.size
    if N < 2:
        return None
    P = float(-2.0 * np.log(pv).sum())
    p_chi = float(stats.chi2.sf(P, 2 * N))
    Z = float(stats.norm.ppf(pv).sum() / np.sqrt(N))
    p_z = float(stats.norm.cdf(Z))
    Pm = float((-2.0 * np.log(pv) - 2.0).sum() / np.sqrt(4.0 * N))
    p_pm = float(stats.norm.sf(Pm))
    return {"test": label, "N": N, "h0": h0,
            "P": f(P), "pP": f(p_chi), "dfP": 2 * N,
            "Z": f(Z), "pZ": f(p_z),
            "Pm": f(Pm), "pPm": f(p_pm),
            "stat": f(P), "p": f(p_chi), "dist": f"χ²({2 * N})",
            "sig": bool(p_chi < 0.05)}


def fisher_adf(series, p, trend):
    pvals = []
    for s in series:
        y = np.asarray(s, float)
        t = _fast_adf_t(y, p, trend)
        pvals.append(mackinnon_adf_pvalue(t, trend=trend))
    return _fisher(pvals, "Fisher-ADF (Maddala-Wu / Choi)",
                   "Tüm birimler için birim kök vardır")


def fisher_pp(series, trend):
    pvals = []
    for s in series:
        _, pv = phillips_perron(np.asarray(s, float), trend=trend)
        pvals.append(pv)
    return _fisher(pvals, "Fisher-PP (Maddala-Wu / Choi)",
                   "Tüm birimler için birim kök vardır")


# --------------------------------------------------------------------------
# Hadri LM
# --------------------------------------------------------------------------

def hadri_test(series, trend, hetero=True):
    """Hadri (2000) LM testi. H0: tum birimler duragan."""
    lms, num, den, used = [], 0.0, 0.0, 0
    per = []
    for s in series:
        y = np.asarray(s, float)
        T = y.size
        if T < 5:
            continue
        if trend == "ct":
            Z = np.column_stack([np.ones(T), np.arange(1, T + 1, dtype=float)])
        else:
            Z = np.ones((T, 1))
        try:
            b = np.linalg.pinv(Z.T @ Z) @ (Z.T @ y)
        except np.linalg.LinAlgError:
            continue
        e = y - Z @ b
        S = np.cumsum(e)
        s2 = float(e @ e) / T
        if s2 <= 0:
            continue
        per.append(float((S ** 2).sum()) / (T ** 2 * s2))
        used += 1
    if used < 2:
        return None
    LM = float(np.mean(per))
    if trend == "ct":
        xi, zeta2 = 1.0 / 15.0, 11.0 / 6300.0
    else:
        xi, zeta2 = 1.0 / 6.0, 1.0 / 45.0
    Z = np.sqrt(used) * (LM - xi) / np.sqrt(zeta2)
    pval = float(stats.norm.sf(Z))
    return {"test": "Hadri LM", "stat": f(Z), "LM": f(LM), "p": f(pval),
            "dist": "N(0,1)", "N": used,
            "h0": "Tüm birimler durağandır (birim kök YOKTUR)",
            "sig": bool(pval < 0.05), "reversed": True}


# --------------------------------------------------------------------------
# Orkestrasyon
# --------------------------------------------------------------------------

def _diff_series(series):
    return [np.diff(np.asarray(s, float)) for s in series if len(s) > 2]


def run_all(pnl: Panel, trend="c", lags="auto", maxlag=2, mc_reps=300,
            tests=("llc", "ips", "fisher_adf", "fisher_pp", "hadri")):
    """Her degisken icin duzey ve birinci farkta panel birim kok testleri."""
    results = []
    for var in pnl.vars:
        raw = [s for _, s in pnl.series(var) if len(s) >= 6]
        if len(raw) < 2:
            continue
        if lags == "auto":
            cand = [bic_lag_select(s, maxlag, trend) for s in raw]
            p = int(np.round(np.median(cand))) if cand else 1
        else:
            p = int(lags)
        p = int(max(0, min(p, maxlag)))

        lvl, dif = {}, {}
        dser = _diff_series(raw)
        pd_ = max(0, p - 1)

        if "llc" in tests:
            lvl["llc"] = llc_test(raw, p, trend, mc_reps)
            dif["llc"] = llc_test(dser, pd_, "c", mc_reps)
        if "ips" in tests:
            lvl["ips"] = ips_test(raw, p, trend)
            dif["ips"] = ips_test(dser, pd_, "c")
        if "fisher_adf" in tests:
            lvl["fisherADF"] = fisher_adf(raw, p, trend)
            dif["fisherADF"] = fisher_adf(dser, pd_, "c")
        if "fisher_pp" in tests:
            lvl["fisherPP"] = fisher_pp(raw, trend)
            dif["fisherPP"] = fisher_pp(dser, "c")
        if "hadri" in tests:
            lvl["hadri"] = hadri_test(raw, trend)
            dif["hadri"] = hadri_test(dser, "c")

        def _votes(block):
            yes = tot = 0
            for key, r in block.items():
                if not r or r.get("p") is None:
                    continue
                tot += 1
                sig = r["p"] < 0.05
                if r.get("reversed"):
                    if not sig:
                        yes += 1
                elif sig:
                    yes += 1
            return yes, tot

        y_l, t_l = _votes(lvl)
        y_d, t_d = _votes(dif)
        stat_level = t_l > 0 and y_l >= (t_l / 2.0)
        stat_diff = t_d > 0 and y_d >= (t_d / 2.0)
        if stat_level:
            order, concl = "I(0)", "Seri düzeyde durağandır."
        elif stat_diff:
            order, concl = "I(1)", "Seri düzeyde durağan değildir; birinci farkında durağan hâle gelmektedir."
        else:
            order, concl = "I(2)?", "Seri birinci farkında da durağanlaşmamıştır; ikinci fark veya alternatif dönüşüm gerekebilir."

        results.append({
            "variable": var, "lags": p, "trend": trend,
            "trendLabel": TREND_LABEL.get(trend, trend),
            "level": lvl, "diff": dif,
            "votesLevel": [y_l, t_l], "votesDiff": [y_d, t_d],
            "order": order, "conclusion": concl,
        })

    orders = {r["variable"]: r["order"] for r in results}
    all_i1 = bool(results) and all(o == "I(1)" for o in orders.values())
    any_i0 = any(o == "I(0)" for o in orders.values())
    return {
        "results": results, "orders": orders,
        "allI1": all_i1, "anyI0": any_i0,
        "options": {"trend": trend, "trendLabel": TREND_LABEL.get(trend, trend),
                    "lags": lags, "maxlag": maxlag, "mcReps": mc_reps},
        "cointegrationAdvised": all_i1,
        "note": ("Tüm seriler I(1) olduğundan eşbütünleşme analizi uygundur."
                 if all_i1 else
                 "Seriler farklı bütünleşme derecelerine sahip; klasik eşbütünleşme "
                 "testleri yerine ARDL sınır testi yaklaşımı değerlendirilmelidir."),
    }
