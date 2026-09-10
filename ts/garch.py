"""Volatilite modellemesi: ARCH-LM testi ve GARCH ailesi modelleri."""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import het_arch

from panel.utils import f, star
from .prepare import TSData

warnings.filterwarnings("ignore")

MODEL_LABEL = {"GARCH": "GARCH", "EGARCH": "EGARCH", "GJR": "GJR-GARCH (TGARCH)"}


def arch_lm(y, nlags=None):
    """Engle (1982) ARCH-LM testi."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 15:
        return None
    e = y - y.mean()
    lags = int(nlags or max(1, min(12, n // 5)))
    try:
        r = het_arch(e, nlags=lags)
    except Exception:
        return None
    return {"name": f"ARCH-LM({lags})", "lags": lags,
            "stat": f(r[0]), "p": f(r[1]),
            "fStat": f(r[2]), "fP": f(r[3]),
            "h0": "Artıklarda ARCH etkisi (koşullu değişen varyans) yoktur",
            "sig": bool(r[1] < 0.05),
            "decision": ("ARCH etkisi vardır; GARCH modellemesi uygundur"
                         if r[1] < 0.05 else
                         "ARCH etkisi yoktur; GARCH modellemesine gerek görülmemektedir")}


def fit_garch(y, model="GARCH", p=1, q=1, o=0, dist="normal", mean="Constant"):
    """arch kutuphanesi ile GARCH ailesi tahmini."""
    try:
        from arch import arch_model
    except ImportError:
        return {"error": "arch kütüphanesi kurulu değil."}

    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size < 40:
        return None

    # olcek: arch kutuphanesi cok kucuk degerlerde uyari verir
    scale = 1.0
    sd = float(np.std(y, ddof=1))
    if 0 < sd < 0.1:
        scale = 100.0

    vol = {"GARCH": "GARCH", "EGARCH": "EGARCH", "GJR": "GARCH"}.get(model, "GARCH")
    o_use = 1 if model == "GJR" else o
    try:
        am = arch_model(y * scale, mean=mean, vol=vol, p=p, o=o_use, q=q, dist=dist)
        res = am.fit(disp="off", show_warning=False)
    except Exception as exc:
        return {"error": str(exc)}

    names = list(res.params.index)
    par = np.asarray(res.params, float)
    se = np.asarray(res.std_err, float)
    tv = np.asarray(res.tvalues, float)
    pv = np.asarray(res.pvalues, float)
    coeffs = [{"name": str(names[i]), "B": f(par[i]), "SE": f(se[i]),
               "t": f(tv[i]), "p": f(pv[i]), "sig": star(pv[i])}
              for i in range(len(names))]

    # kaliciligi hesapla (alpha + beta)
    a = sum(par[i] for i, nm in enumerate(names) if str(nm).startswith("alpha"))
    b = sum(par[i] for i, nm in enumerate(names) if str(nm).startswith("beta"))
    g = sum(par[i] for i, nm in enumerate(names) if str(nm).startswith("gamma"))
    persist = float(a + b + (0.5 * g if model == "GJR" else 0.0))

    cv = np.asarray(res.conditional_volatility, float) / scale
    std_resid = np.asarray(res.std_resid, float)

    lb = None
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lag = int(min(10, max(4, std_resid.size // 5)))
        t = acorr_ljungbox(std_resid ** 2, lags=[lag], return_df=True)
        lb = {"name": f"Ljung-Box Q²({lag})",
              "stat": f(float(t["lb_stat"].iloc[0])),
              "p": f(float(t["lb_pvalue"].iloc[0])),
              "h0": "Standartlaştırılmış kare artıklarda otokorelasyon yoktur",
              "ok": bool(float(t["lb_pvalue"].iloc[0]) > 0.05)}
    except Exception:
        pass

    post = None
    try:
        post = arch_lm(std_resid)
    except Exception:
        pass

    return {
        "model": f"{MODEL_LABEL.get(model, model)}({p},{q})"
                 + ("" if model != "GJR" else " [asimetrik]"),
        "family": model, "p": p, "q": q, "dist": dist,
        "coeffs": coeffs,
        "persistence": f(persist),
        "stationary": bool(persist < 1.0),
        "halfLife": f(np.log(0.5) / np.log(persist)) if 0 < persist < 1 else None,
        "fit": {"aic": f(float(res.aic)), "bic": f(float(res.bic)),
                "logLik": f(float(res.loglikelihood)), "nobs": int(res.nobs)},
        "condVol": [f(x, 6) for x in cv],
        "stdResid": [f(x, 6) for x in std_resid],
        "postArch": post, "ljungBoxSq": lb,
        "scaled": bool(scale != 1.0), "scale": scale,
    }


def run(ts: TSData, var, models=("GARCH", "GJR", "EGARCH"), p=1, q=1,
        dist="normal", use_returns=False):
    y = ts.y(var)
    labels = list(ts.labels)
    if use_returns:
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.diff(y) / y[:-1] * 100.0
        y = y[np.isfinite(y)]
        labels = labels[1:]
        series_label = f"{var} getiri serisi (%)"
    else:
        series_label = var

    test = arch_lm(y)
    out = {"variable": var, "seriesLabel": series_label,
           "useReturns": bool(use_returns), "archLM": test,
           "labels": labels, "series": [f(x, 6) for x in y]}

    # ARCH-LM anlamsız çıksa bile modüller kullanıcı tarafından açıkça seçildiği
    # için modeller yine tahmin edilir; test sonucu yorum notu olarak verilir.
    fits = []
    for m in models:
        r = fit_garch(y, model=m, p=p, q=q, dist=dist)
        if r and "error" not in r:
            fits.append(r)

    if not fits:
        out["models"] = []
        out["note"] = ("GARCH modelleri tahmin edilemedi; volatilite modellemesi "
                       "için en az 40 gözlem gereklidir.")
        return out

    fits.sort(key=lambda r: (r["fit"]["aic"] if r["fit"]["aic"] is not None else 1e18))
    out["models"] = fits
    out["best"] = fits[0]
    note = f"AIC ölçütüne göre en uygun model {fits[0]['model']} olarak belirlenmiştir."
    if test and not test.get("sig"):
        note += (" Ancak ARCH-LM testi %5 düzeyinde anlamlı çıkmadığından "
                 "serideki koşullu değişen varyans zayıftır; GARCH sonuçları "
                 "dikkatle yorumlanmalıdır.")
    out["note"] = note
    out["archSupported"] = bool(test and test.get("sig"))
    return out
