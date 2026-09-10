"""VAR modeli: gecikme secimi, nedensellik, etki-tepki ve varyans ayristirmasi."""
from __future__ import annotations

import warnings
from itertools import permutations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.api import VAR

from panel.utils import f, ols, star
from .prepare import TSData

warnings.filterwarnings("ignore")

IC_LABEL = {"aic": "Akaike (AIC)", "bic": "Schwarz (BIC/SC)",
            "hqic": "Hannan-Quinn (HQ)", "fpe": "Son Öngörü Hatası (FPE)"}


# --------------------------------------------------------------------------

def select_order(D, maxlags=8):
    try:
        sel = VAR(D).select_order(maxlags=maxlags)
    except Exception:
        return None, None
    tbl = []
    try:
        for lag in range(0, maxlags + 1):
            row = {"lag": lag}
            for ic in ("aic", "bic", "hqic", "fpe"):
                arr = getattr(sel.ics, ic, None)
                row[ic] = f(float(arr[lag])) if arr is not None and lag < len(arr) else None
            tbl.append(row)
    except Exception:
        tbl = []
    chosen = {k: int(v) for k, v in (sel.selected_orders or {}).items()}
    return sel, {"table": tbl, "selected": chosen,
                 "labels": {k: IC_LABEL.get(k, k) for k in IC_LABEL}}


def _ty_design(D, names, total_lags):
    """Toda-Yamamoto icin gecikmeli tasarim matrisi."""
    T, k = D.shape
    n = T - total_lags
    if n < k * total_lags + 5:
        return None, None, None
    Y = D[total_lags:, :]
    cols = [np.ones(n)]
    idx = {}
    pos = 1
    for L in range(1, total_lags + 1):
        for j in range(k):
            cols.append(D[total_lags - L: T - L, j])
            idx[(j, L)] = pos
            pos += 1
    return Y, np.column_stack(cols), idx


def toda_yamamoto(D, names, k_lags, dmax):
    """Toda & Yamamoto (1995) gecikme genisletmeli nedensellik testi."""
    total = int(k_lags + dmax)
    Y, Z, idx = _ty_design(D, names, total)
    if Y is None:
        return []
    out = []
    for ci, cause in enumerate(names):
        for ei, effect in enumerate(names):
            if ci == ei:
                continue
            try:
                r = ols(Y[:, ei], Z, has_const=True)
            except Exception:
                continue
            rows = [idx[(ci, L)] for L in range(1, k_lags + 1) if (ci, L) in idx]
            if not rows:
                continue
            R = np.zeros((len(rows), Z.shape[1]))
            for a, b in enumerate(rows):
                R[a, b] = 1.0
            Rb = R @ r.beta
            M = R @ r.cov @ R.T
            try:
                w = float(Rb @ np.linalg.pinv(M) @ Rb)
            except np.linalg.LinAlgError:
                continue
            dfree = len(rows)
            p = float(stats.chi2.sf(w, dfree))
            out.append({
                "cause": cause, "effect": effect,
                "h0": f"{cause}, {effect} değişkeninin Granger nedeni değildir",
                "stat": f(w), "df": dfree, "p": f(p), "star": star(p),
                "sig": bool(p < 0.05),
                "decision": "H₀ reddedilir" if p < 0.05 else "H₀ reddedilemez",
            })
    return out


def granger_var(res, names):
    out = []
    for cause, effect in permutations(names, 2):
        try:
            t = res.test_causality(effect, [cause], kind="f")
        except Exception:
            continue
        p = float(t.pvalue)
        out.append({
            "cause": cause, "effect": effect,
            "h0": f"{cause}, {effect} değişkeninin Granger nedeni değildir",
            "stat": f(float(t.test_statistic)), "df": str(t.df),
            "p": f(p), "star": star(p), "sig": bool(p < 0.05),
            "decision": "H₀ reddedilir" if p < 0.05 else "H₀ reddedilemez",
        })
    return out


def _summary(pairs, names):
    seen, summary = set(), []
    look = {(x["cause"], x["effect"]): x for x in pairs}
    for a, b in permutations(names, 2):
        if (b, a) in seen:
            continue
        seen.add((a, b))
        f1, f2 = look.get((a, b)), look.get((b, a))
        if not f1 or not f2:
            continue
        s1, s2 = f1["sig"], f2["sig"]
        if s1 and s2:
            rel, arrow = "Çift yönlü nedensellik", f"{a} ↔ {b}"
        elif s1:
            rel, arrow = "Tek yönlü nedensellik", f"{a} → {b}"
        elif s2:
            rel, arrow = "Tek yönlü nedensellik", f"{b} → {a}"
        else:
            rel, arrow = "Nedensellik yok", f"{a} ✕ {b}"
        summary.append({"pair": f"{a} – {b}", "relation": rel, "arrow": arrow,
                        "p1": f1["p"], "p2": f2["p"]})
    return summary


def run(ts: TSData, variables=None, maxlags=8, ic="aic", lags=None,
        irf_periods=10, use_diff=None, dmax=None):
    names = list(variables or ts.vars)
    if len(names) < 2:
        return {"available": False,
                "note": "VAR analizi için en az iki değişken gereklidir."}

    D0 = ts.matrix(names)
    labels = list(ts.labels)
    if use_diff:
        D = np.diff(D0, axis=0)
        labels = labels[1:]
        disp = [f"Δ{v}" for v in names]
    else:
        D, disp = D0, names

    if D.shape[0] < 20:
        return {"available": False,
                "note": "VAR analizi için gözlem sayısı yetersizdir (en az 20)."}

    maxlags = int(min(maxlags, max(1, (D.shape[0] - 5) // (len(names) + 1))))
    sel, sel_tbl = select_order(D, maxlags)
    k = int(lags) if lags else int((sel_tbl or {}).get("selected", {}).get(ic, 1) or 1)
    k = max(1, min(k, maxlags))

    try:
        res = VAR(pd.DataFrame(D, columns=disp)).fit(k)
    except Exception as exc:
        return {"available": False, "note": f"VAR tahmin edilemedi: {exc}"}

    # --- kararlilik
    try:
        roots = np.abs(np.asarray(res.roots, float))
        stable = bool(res.is_stable(verbose=False))
    except Exception:
        roots, stable = np.array([]), None

    # --- artik tanilari
    diag = {}
    try:
        w = res.test_whiteness(nlags=min(k + 6, D.shape[0] // 3), adjusted=True)
        diag["whiteness"] = {"name": "Portmanteau (LM) otokorelasyon testi",
                             "h0": "Artıklarda otokorelasyon yoktur",
                             "stat": f(float(w.test_statistic)), "p": f(float(w.pvalue)),
                             "ok": bool(float(w.pvalue) > 0.05)}
    except Exception:
        pass
    try:
        nt = res.test_normality()
        diag["normality"] = {"name": "Jarque-Bera (çok değişkenli)",
                             "h0": "Artıklar normal dağılmaktadır",
                             "stat": f(float(nt.test_statistic)), "p": f(float(nt.pvalue)),
                             "ok": bool(float(nt.pvalue) > 0.05)}
    except Exception:
        pass

    # --- denklem katsayilari
    eqs = []
    try:
        prm = res.params
        bse = res.stderr
        tv = res.tvalues
        pv = res.pvalues
        for col in prm.columns:
            eqs.append({
                "equation": str(col),
                "coeffs": [{"name": str(ix), "B": f(float(prm.loc[ix, col])),
                            "SE": f(float(bse.loc[ix, col])),
                            "t": f(float(tv.loc[ix, col])),
                            "p": f(float(pv.loc[ix, col])),
                            "sig": star(float(pv.loc[ix, col]))}
                           for ix in prm.index],
            })
    except Exception:
        pass

    # --- etki-tepki
    irf_out = None
    try:
        periods = int(min(irf_periods, max(4, D.shape[0] // 3)))
        ir = res.irf(periods)
        eff = np.asarray(ir.irfs, float)          # (periods+1, k, k)
        try:
            se = np.asarray(ir.stderr(), float)
        except Exception:
            se = np.zeros_like(eff)
        cum = np.asarray(ir.cum_effects, float)
        panels = []
        for i, shock in enumerate(disp):          # şok kaynağı
            for j, resp in enumerate(disp):       # tepki veren
                m = eff[:, j, i]
                s = se[:, j, i] if se.shape == eff.shape else np.zeros_like(m)
                panels.append({
                    "shock": shock, "response": resp,
                    "values": [f(x, 6) for x in m],
                    "lower": [f(x - 1.96 * y, 6) for x, y in zip(m, s)],
                    "upper": [f(x + 1.96 * y, 6) for x, y in zip(m, s)],
                    "cumulative": [f(x, 6) for x in cum[:, j, i]],
                })
        irf_out = {"periods": periods, "panels": panels,
                   "steps": list(range(periods + 1))}
    except Exception:
        irf_out = None

    # --- varyans ayristirmasi
    fevd_out = None
    try:
        periods = int(min(irf_periods, max(4, D.shape[0] // 3)))
        fe = res.fevd(periods)
        dec = np.asarray(fe.decomp, float)        # (k, periods, k)
        tables = []
        for j, resp in enumerate(disp):
            rows = []
            for t in range(periods):
                rows.append({"period": t + 1,
                             "shares": [f(100 * dec[j, t, i], 2) for i in range(len(disp))]})
            tables.append({"variable": resp, "sources": disp, "rows": rows})
        fevd_out = {"periods": periods, "tables": tables}
    except Exception:
        fevd_out = None

    # --- nedensellik
    gr = granger_var(res, disp)
    ty = []
    if not use_diff:
        dmax_use = int(dmax if dmax is not None else 1)
        try:
            ty = toda_yamamoto(D0, names, k, dmax_use)
        except Exception:
            ty = []

    return {
        "available": True,
        "vars": disp, "originalVars": names, "useDiff": bool(use_diff),
        "lags": k, "ic": ic, "icLabel": IC_LABEL.get(ic, ic),
        "lagSelection": sel_tbl,
        "nobs": int(res.nobs),
        # statsmodels karakteristik polinomun koklerini dondurur; kararlilik
        # kosulu |kok| > 1'dir. Yaygin gosterim ters kokler (1/|kok|) uzerinden
        # yapildigi icin her ikisi de raporlanir.
        "stability": {
            "stable": stable,
            "minRoot": f(float(np.min(roots))) if roots.size else None,
            "maxInverseRoot": f(float(1.0 / np.min(roots)))
            if roots.size and np.min(roots) > 0 else None,
            "inverseRoots": [f(1.0 / x, 4) for x in roots[:12] if x > 0],
            "note": ("Karakteristik polinomun tüm ters kökleri birim çember "
                     "içinde yer almaktadır; VAR modeli kararlılık koşulunu "
                     "sağlamaktadır." if stable else
                     "En az bir ters kök birim çember dışında kalmaktadır; model "
                     "kararlılık koşulunu sağlamamaktadır, gecikme uzunluğu veya "
                     "değişken seti gözden geçirilmelidir.")},
        "diagnostics": diag,
        "equations": eqs,
        "granger": gr, "grangerSummary": _summary(gr, disp),
        "todaYamamoto": ty,
        "todaYamamotoSummary": _summary(ty, names) if ty else [],
        "dmax": int(dmax if dmax is not None else 1),
        "irf": irf_out, "fevd": fevd_out,
        "aic": f(float(res.aic)), "bic": f(float(res.bic)),
        "hqic": f(float(res.hqic)), "logLik": f(float(res.llf)),
    }
