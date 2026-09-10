"""Birim kok testleri: ADF, Phillips-Perron, KPSS ve Zivot-Andrews."""
from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews

from panel.utils import f, phillips_perron
from .prepare import TSData

warnings.filterwarnings("ignore")

TREND_LABEL = {"n": "Sabitsiz ve trendsiz", "c": "Sabitli",
               "ct": "Sabitli ve trendli"}
ZA_REGRESSION = {"c": "Sabitte kırılma", "t": "Trendde kırılma",
                 "ct": "Sabit ve trendde kırılma"}


def _crit(d):
    return {k.replace("%", ""): f(v) for k, v in (d or {}).items()}


def adf_test(y, trend="c", maxlag=None, autolag="AIC"):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size < 8:
        return None
    try:
        r = adfuller(y, maxlag=maxlag, regression=trend, autolag=autolag)
    except Exception:
        return None
    return {"test": "ADF", "stat": f(r[0]), "p": f(r[1]), "lags": int(r[2]),
            "nobs": int(r[3]), "crit": _crit(r[4]),
            "h0": "Seri birim kök içermektedir (durağan değildir)",
            "sig": bool(r[1] < 0.05)}


def pp_test(y, trend="c"):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size < 8:
        return None
    stat, p = phillips_perron(y, trend=trend)
    if stat is None or not np.isfinite(stat):
        return None
    # ADF ile ayni asimptotik kritik degerler
    try:
        ref = adfuller(y, maxlag=0, regression=trend, autolag=None)
        crit = _crit(ref[4])
    except Exception:
        crit = {}
    return {"test": "PP", "stat": f(stat), "p": f(p), "crit": crit,
            "h0": "Seri birim kök içermektedir (durağan değildir)",
            "sig": bool(p is not None and p < 0.05)}


def kpss_test(y, trend="c", nlags="auto"):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size < 8:
        return None
    reg = "ct" if trend == "ct" else "c"
    try:
        stat, p, lags, crit = kpss(y, regression=reg, nlags=nlags)
    except Exception:
        return None
    return {"test": "KPSS", "stat": f(stat), "p": f(p), "lags": int(lags),
            "crit": _crit(crit),
            "h0": "Seri durağandır (birim kök YOKTUR)",
            "sig": bool(p < 0.05), "reversed": True}


def za_test(y, labels=None, regression="ct", maxlag=None):
    """Zivot-Andrews icsel yapisal kirilmali birim kok testi."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size < 20:
        return None
    try:
        stat, p, crit, baselag, bpidx = zivot_andrews(
            y, regression=regression, maxlag=maxlag)
    except Exception:
        return None
    lab = None
    if labels is not None and 0 <= int(bpidx) < len(labels):
        lab = str(labels[int(bpidx)])
    return {"test": "Zivot-Andrews", "stat": f(stat), "p": f(p),
            "crit": _crit(crit), "lags": int(baselag),
            "breakIndex": int(bpidx), "breakLabel": lab,
            "regression": regression,
            "regressionLabel": ZA_REGRESSION.get(regression, regression),
            "h0": "Seri, yapısal kırılma altında birim kök içermektedir",
            "sig": bool(p < 0.05)}


# --------------------------------------------------------------------------

def _order(level_votes, diff_votes):
    y_l, t_l = level_votes
    y_d, t_d = diff_votes
    stat_l = t_l > 0 and y_l >= t_l / 2.0
    stat_d = t_d > 0 and y_d >= t_d / 2.0
    if stat_l:
        return "I(0)", "Seri düzeyde durağandır."
    if stat_d:
        return "I(1)", ("Seri düzeyde durağan değildir; birinci farkı "
                        "alındığında durağan hâle gelmektedir.")
    return "I(2)?", ("Seri birinci farkında da durağanlaşmamıştır; ikinci fark "
                     "veya logaritmik dönüşüm değerlendirilmelidir.")


def _votes(block):
    yes = tot = 0
    for r in block.values():
        if not r or r.get("p") is None:
            continue
        tot += 1
        sig = r["p"] < 0.05
        if r.get("reversed"):          # KPSS: H0 duraganlik
            if not sig:
                yes += 1
        elif sig:
            yes += 1
    return yes, tot


def run_all(ts: TSData, trend="c", maxlag=None, za_regression="ct",
            tests=("adf", "pp", "kpss", "za")):
    results = []
    for v in ts.vars:
        y = ts.y(v)
        dy = np.diff(y)
        lvl, dif = {}, {}
        if "adf" in tests:
            lvl["adf"] = adf_test(y, trend, maxlag)
            dif["adf"] = adf_test(dy, "c", maxlag)
        if "pp" in tests:
            lvl["pp"] = pp_test(y, trend)
            dif["pp"] = pp_test(dy, "c")
        if "kpss" in tests:
            lvl["kpss"] = kpss_test(y, trend)
            dif["kpss"] = kpss_test(dy, "c")
        if "za" in tests:
            lvl["za"] = za_test(y, ts.labels, za_regression, maxlag)
            dif["za"] = za_test(dy, ts.labels[1:], "c", maxlag)

        order, concl = _order(_votes(lvl), _votes(dif))
        results.append({
            "variable": v, "level": lvl, "diff": dif,
            "order": order, "conclusion": concl,
            "votesLevel": list(_votes(lvl)), "votesDiff": list(_votes(dif)),
            "breakLabel": (lvl.get("za") or {}).get("breakLabel"),
        })

    orders = {r["variable"]: r["order"] for r in results}
    vals = list(orders.values())
    all_i1 = bool(vals) and all(o == "I(1)" for o in vals)
    mixed = bool(vals) and len(set(vals)) > 1
    any_i2 = any(o.startswith("I(2)") for o in vals)

    if all_i1:
        note = ("Tüm seriler I(1) olduğundan Johansen ve Engle-Granger "
                "eşbütünleşme testleri uygulanabilir.")
    elif any_i2:
        note = ("En az bir seri I(2) görünmektedir; klasik eşbütünleşme testleri "
                "geçerli değildir, logaritmik dönüşüm veya ikinci fark "
                "değerlendirilmelidir.")
    elif mixed:
        note = ("Seriler farklı bütünleşme derecelerine sahiptir (I(0) ve I(1) "
                "karışık); bu durumda ARDL sınır testi yaklaşımı uygundur.")
    else:
        note = ("Tüm seriler düzeyde durağandır; doğrudan VAR modeli veya klasik "
                "regresyon yaklaşımı kullanılabilir.")

    return {
        "results": results, "orders": orders,
        "allI1": all_i1, "mixed": mixed, "anyI2": any_i2,
        "cointegrationAdvised": all_i1,
        "ardlAdvised": bool(mixed and not any_i2),
        "note": note,
        "options": {"trend": trend, "trendLabel": TREND_LABEL.get(trend, trend),
                    "maxlag": maxlag,
                    "zaRegression": ZA_REGRESSION.get(za_regression, za_regression)},
    }
