"""Esbutunlesme: Engle-Granger, Johansen, VECM ve ARDL sinir testi."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.ardl import ARDL, UECM
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

from panel.utils import add_const, f, ols, star
from .prepare import TSData

warnings.filterwarnings("ignore")

DET_LABEL = {"n": "Deterministik terim yok", "co": "Eşbütünleşme denkleminde sabit",
             "ci": "Eşbütünleşme denkleminde sabit", "lo": "Düzeyde sabit",
             "cili": "Sabit ve trend"}
JOHANSEN_DET = {0: "Sabitli (eşbütünleşme denkleminde)",
                1: "Sabitli ve trendli", -1: "Deterministik terim yok"}
ARDL_CASE = {1: "Durum I: sabit ve trend yok", 2: "Durum II: kısıtlı sabit",
             3: "Durum III: kısıtsız sabit, trend yok",
             4: "Durum IV: kısıtsız sabit, kısıtlı trend",
             5: "Durum V: kısıtsız sabit ve trend"}


# ==========================================================================
# Engle-Granger
# ==========================================================================

def engle_granger(ts: TSData, dep, indep, trend="c", maxlag=None):
    y = ts.y(dep)
    X = ts.matrix(indep)
    if y.size < 15:
        return None
    try:
        stat, pval, crit = coint(y, X, trend=trend, maxlag=maxlag, autolag="AIC")
    except Exception:
        return None

    # uzun donem denklemi ve artiklar
    Z = add_const(X)
    r = ols(y, Z)
    resid = r.resid
    names = ["Sabit"] + list(indep)
    lr = [{"name": names[i], "B": f(r.beta[i]), "SE": f(r.se[i]),
           "t": f(r.tstat[i]), "p": f(r.pval[i]), "sig": star(r.pval[i])}
          for i in range(len(names))]

    # hata duzeltme modeli:  dY_t = a + phi*ECT_{t-1} + b*dX_t + e
    dy = np.diff(y)
    dX = np.diff(X, axis=0)
    ect = resid[:-1]
    Zc = np.column_stack([np.ones(dy.size), ect, dX])
    ecm = None
    if Zc.shape[0] > Zc.shape[1] + 2:
        re = ols(dy, Zc)
        enames = ["Sabit", "ECT(-1)"] + [f"Δ{v}" for v in indep]
        ecm = {
            "coeffs": [{"name": enames[i], "B": f(re.beta[i]), "SE": f(re.se[i]),
                        "t": f(re.tstat[i]), "p": f(re.pval[i]),
                        "sig": star(re.pval[i])} for i in range(len(enames))],
            "r2": f(re.r2), "F": f(re.fstat), "pF": f(re.fpval),
            "speed": f(re.beta[1]), "speedP": f(re.pval[1]),
            "valid": bool(re.beta[1] < 0 and re.pval[1] < 0.05),
            "halfLife": f(np.log(0.5) / np.log(1 + re.beta[1]))
            if -1 < re.beta[1] < 0 else None,
        }

    return {
        "name": "Engle-Granger Eşbütünleşme Testi",
        "h0": "Eşbütünleşme ilişkisi yoktur (artıklar durağan değildir)",
        "dep": dep, "indep": list(indep),
        "stat": f(stat), "p": f(pval),
        "crit": {"1": f(crit[0]), "5": f(crit[1]), "10": f(crit[2])},
        "sig": bool(pval < 0.05),
        "decision": "Eşbütünleşme vardır" if pval < 0.05 else "Eşbütünleşme yoktur",
        "longRun": lr, "r2": f(r.r2), "ecm": ecm,
        "residuals": [f(x, 6) for x in resid],
        "residLabels": ts.labels,
    }


# ==========================================================================
# Johansen
# ==========================================================================

def johansen(ts: TSData, variables, det_order=0, k_ar_diff=1):
    if len(variables) < 2:
        return None
    D = ts.matrix(variables)
    if D.shape[0] < 4 * k_ar_diff + 12:
        return None
    try:
        j = coint_johansen(D, det_order, k_ar_diff)
    except Exception:
        return None

    k = len(variables)
    trace, maxeig = [], []
    rank_trace = rank_max = 0
    for i in range(k):
        h0 = f"r = {i}" if i == 0 else f"r ≤ {i}"
        t_sig = bool(j.lr1[i] > j.cvt[i, 1])
        m_sig = bool(j.lr2[i] > j.cvm[i, 1])
        if t_sig:
            rank_trace = i + 1
        if m_sig:
            rank_max = i + 1
        trace.append({"h0": h0, "eig": f(j.eig[i], 4), "stat": f(j.lr1[i]),
                      "c90": f(j.cvt[i, 0]), "c95": f(j.cvt[i, 1]),
                      "c99": f(j.cvt[i, 2]), "sig": t_sig,
                      "star": "**" if t_sig else ""})
        maxeig.append({"h0": h0, "eig": f(j.eig[i], 4), "stat": f(j.lr2[i]),
                       "c90": f(j.cvm[i, 0]), "c95": f(j.cvm[i, 1]),
                       "c99": f(j.cvm[i, 2]), "sig": m_sig,
                       "star": "**" if m_sig else ""})

    rank = min(rank_trace, rank_max) if (rank_trace and rank_max) else max(rank_trace, rank_max)
    return {
        "name": "Johansen Eşbütünleşme Testi",
        "vars": list(variables), "detOrder": det_order,
        "detLabel": JOHANSEN_DET.get(det_order, str(det_order)),
        "kArDiff": int(k_ar_diff),
        "trace": trace, "maxEigen": maxeig,
        "rankTrace": int(rank_trace), "rankMaxEigen": int(rank_max),
        "rank": int(rank),
        "cointegrated": bool(rank >= 1),
        "decision": (f"{rank} adet eşbütünleşme vektörü bulunmaktadır"
                     if rank >= 1 else "Eşbütünleşme ilişkisi bulunmamaktadır"),
    }


def vecm(ts: TSData, variables, rank=1, k_ar_diff=1, deterministic="ci"):
    if len(variables) < 2 or rank < 1:
        return None
    D = pd.DataFrame(ts.matrix(variables), columns=list(variables))
    try:
        res = VECM(D, k_ar_diff=k_ar_diff, coint_rank=rank,
                   deterministic=deterministic).fit()
    except Exception:
        return None

    alpha = np.asarray(res.alpha, float)
    beta = np.asarray(res.beta, float)
    pa = np.asarray(res.pvalues_alpha, float)
    ta = np.asarray(res.tvalues_alpha, float)
    se_a = np.asarray(res.stderr_alpha, float)

    adj = []
    for i, v in enumerate(variables):
        adj.append({"variable": v, "alpha": f(alpha[i, 0]), "SE": f(se_a[i, 0]),
                    "t": f(ta[i, 0]), "p": f(pa[i, 0]), "sig": star(pa[i, 0]),
                    "valid": bool(alpha[i, 0] < 0 and pa[i, 0] < 0.05)})

    # --- esbutunlesme vektorleri (beta matrisi, r sutun)
    #     Normalizasyon rank > 1 iken tek turlu degildir; bu nedenle tum
    #     vektorler tablo olarak verilir ve tek denklem gosterimi yalnizca
    #     rank = 1 oldugunda uretilir.
    vectors = []
    for c in range(beta.shape[1]):
        b = beta[:, c].astype(float)
        piv = 0 if abs(b[0]) > 1e-8 else int(np.argmax(np.abs(b)))
        scale = b[piv] if abs(b[piv]) > 1e-12 else 1.0
        nb = b / scale
        vectors.append({
            "index": c + 1,
            "normalizedOn": variables[piv],
            "coeffs": [{"variable": variables[i], "beta": f(nb[i])}
                       for i in range(len(variables))],
        })

    coint_vec = vectors[0]["coeffs"]
    eq = None
    if beta.shape[1] == 1:
        nb = [c["beta"] for c in coint_vec]
        piv = vectors[0]["normalizedOn"]
        terms = []
        for i, v in enumerate(variables):
            if v == piv:
                continue
            val = -(nb[i] if nb[i] is not None else 0.0)
            terms.append(f"{'+' if val >= 0 else '−'} {abs(val):.4f}".replace(".", ",")
                         + f"·{v}")
        eq = f"{piv} = " + " ".join(terms) if terms else None

    speed = alpha[0, 0]
    return {
        "name": "Vektör Hata Düzeltme Modeli (VECM)",
        "vars": list(variables), "rank": int(rank), "kArDiff": int(k_ar_diff),
        "adjustment": adj, "cointVector": coint_vec, "vectors": vectors,
        "equation": eq,
        "normalizationNote": (None if beta.shape[1] == 1 else
                              "Eşbütünleşme derecesi birden büyük olduğundan "
                              "normalizasyon tek türlü değildir; vektörler ayrı "
                              "ayrı raporlanmıştır."),
        "speed": f(speed), "speedP": f(pa[0, 0]),
        "valid": bool(speed < 0 and pa[0, 0] < 0.05),
        "halfLife": f(np.log(0.5) / np.log(1 + speed)) if -1 < speed < 0 else None,
        "logLik": f(getattr(res, "llf", np.nan)),
    }


# ==========================================================================
# ARDL sinir testi
# ==========================================================================

def ardl_bounds(ts: TSData, dep, indep, max_p=4, max_q=4, case=3, trend="c"):
    y = pd.Series(ts.y(dep), name=dep)
    X = pd.DataFrame({v: ts.y(v) for v in indep})
    if y.size < 20:
        return None

    best, grid = None, []
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:
                r = ARDL(y, p, X, q, trend=trend).fit()
            except Exception:
                continue
            aic = float(r.aic)
            if not np.isfinite(aic):
                continue
            grid.append({"p": p, "q": q, "aic": f(aic), "bic": f(float(r.bic))})
            if best is None or aic < best[0]:
                best = (aic, p, q, r)
    if best is None:
        return None
    _, p, q, ardl_res = best

    try:
        u = UECM.from_ardl(ardl_res.model).fit()
        bt = u.bounds_test(case=case)
    except Exception:
        return None

    cv = bt.critical_values
    stat = float(bt.statistic)
    try:
        lo95 = float(cv.loc[95.0, "lower"]); up95 = float(cv.loc[95.0, "upper"])
    except Exception:
        lo95 = up95 = np.nan
    if np.isfinite(up95) and stat > up95:
        verdict = "Eşbütünleşme vardır (üst sınırın üzerinde)"
        coint_ok = True
    elif np.isfinite(lo95) and stat < lo95:
        verdict = "Eşbütünleşme yoktur (alt sınırın altında)"
        coint_ok = False
    else:
        verdict = "Karar verilemiyor (sınırlar arasında)"
        coint_ok = None

    crit_rows = []
    for pct in cv.index:
        crit_rows.append({"level": f"%{100 - float(pct):.0f}".replace(".0", ""),
                          "percentile": f(float(pct), 1),
                          "lower": f(float(cv.loc[pct, "lower"])),
                          "upper": f(float(cv.loc[pct, "upper"]))})

    # uzun donem katsayilari (esbutunlesme vektoru)
    lr = []
    try:
        cp, cb, ct_, cpv = u.ci_params, u.ci_bse, u.ci_tvalues, u.ci_pvalues
        for nm in cp.index:
            if str(nm) == dep:
                continue
            b = float(cp[nm])
            coefficient = -b if str(nm) != "const" else b
            lr.append({"name": "Sabit" if str(nm) == "const" else str(nm),
                       "B": f(coefficient), "SE": f(float(cb[nm])),
                       "t": f(float(ct_[nm])), "p": f(float(cpv[nm])),
                       "sig": star(float(cpv[nm]))})
    except Exception:
        pass

    # kisa donem + hata duzeltme terimi
    sr, ect = [], None
    try:
        names = list(u.params.index)
        for i, nm in enumerate(names):
            row = {"name": str(nm), "B": f(float(u.params.iloc[i])),
                   "SE": f(float(u.bse.iloc[i])), "t": f(float(u.tvalues.iloc[i])),
                   "p": f(float(u.pvalues.iloc[i])),
                   "sig": star(float(u.pvalues.iloc[i]))}
            if str(nm) == f"{dep}.L1":
                row["name"] = "ECT(-1)"
                ect = row
            sr.append(row)
    except Exception:
        pass

    return {
        "name": "ARDL Sınır Testi (Pesaran, Shin & Smith, 2001)",
        "h0": "Değişkenler arasında uzun dönemli ilişki (eşbütünleşme) yoktur",
        "dep": dep, "indep": list(indep),
        "order": {"p": p, "q": q},
        "orderLabel": f"ARDL({p}, {', '.join([str(q)] * len(indep))})",
        "case": case, "caseLabel": ARDL_CASE.get(case, str(case)),
        "stat": f(stat), "statName": "F-istatistiği",
        "crit": crit_rows,
        "lower95": f(lo95), "upper95": f(up95),
        "cointegrated": coint_ok, "decision": verdict,
        "longRun": lr, "shortRun": sr, "ect": ect,
        "speed": ect["B"] if ect else None,
        "valid": bool(ect and ect["B"] is not None and ect["B"] < 0
                      and ect["p"] is not None and ect["p"] < 0.05),
        "halfLife": (f(np.log(0.5) / np.log(1 + ect["B"]))
                     if ect and ect["B"] is not None and -1 < ect["B"] < 0 else None),
        "aic": f(float(ardl_res.aic)), "bic": f(float(ardl_res.bic)),
        "gridTop": sorted(grid, key=lambda g: g["aic"])[:5],
    }


# ==========================================================================

def run_all(ts: TSData, dep=None, indep=None, det_order=0, k_ar_diff=1,
            case=3, max_p=4, max_q=4, do_ardl=True, do_johansen=True,
            do_eg=True):
    variables = ts.vars
    if len(variables) < 2:
        return {"available": False,
                "note": "Eşbütünleşme analizi için en az iki değişken gereklidir."}
    dep = dep or variables[0]
    indep = indep or [v for v in variables if v != dep]

    out = {"available": True, "dep": dep, "indep": indep}
    if do_eg:
        try:
            out["engleGranger"] = engle_granger(ts, dep, indep)
        except Exception as exc:
            out["engleGrangerError"] = str(exc)
    if do_johansen:
        try:
            jo = johansen(ts, variables, det_order, k_ar_diff)
            out["johansen"] = jo
            if jo and jo["rank"] >= 1:
                out["vecm"] = vecm(ts, variables, jo["rank"], k_ar_diff)
        except Exception as exc:
            out["johansenError"] = str(exc)
    if do_ardl:
        try:
            out["ardl"] = ardl_bounds(ts, dep, indep, max_p, max_q, case)
        except Exception as exc:
            out["ardlError"] = str(exc)

    votes = []
    if out.get("engleGranger"):
        votes.append(bool(out["engleGranger"]["sig"]))
    if out.get("johansen"):
        votes.append(bool(out["johansen"]["cointegrated"]))
    if out.get("ardl") and out["ardl"].get("cointegrated") is not None:
        votes.append(bool(out["ardl"]["cointegrated"]))
    if votes:
        yes = sum(votes)
        out["overall"] = {"yes": yes, "total": len(votes),
                          "cointegrated": yes >= (len(votes) / 2.0),
                          "text": f"{len(votes)} testten {yes} tanesi eşbütünleşme "
                                  "bulgusunu desteklemektedir."}
    return out
