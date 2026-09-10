"""Panel nedensellik: Dumitrescu-Hurlin ve havuzlanmis Granger testleri."""
from __future__ import annotations

from itertools import permutations

import numpy as np
from scipy import stats

from .prepare import Panel
from .utils import f, ols, star


def _unit_wald(y, x, K):
    """
    Birim duzeyinde Granger regresyonu:
      y_t = a + sum_k g_k y_{t-k} + sum_k b_k x_{t-k} + e_t
    H0: tum b_k = 0. Donen: Wald istatistigi (= K * F).
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    T = y.size
    if T < 2 * K + 5:
        return None
    idx = np.arange(K, T)
    cols = [np.ones(idx.size)]
    for k in range(1, K + 1):
        cols.append(y[idx - k])
    xstart = len(cols)
    for k in range(1, K + 1):
        cols.append(x[idx - k])
    Z = np.column_stack(cols)
    if Z.shape[0] <= Z.shape[1] + 2:
        return None
    try:
        r = ols(y[idx], Z, has_const=True)
    except Exception:
        return None
    R = np.zeros((K, Z.shape[1]))
    for k in range(K):
        R[k, xstart + k] = 1.0
    Rb = R @ r.beta
    M = R @ r.cov @ R.T
    try:
        Mi = np.linalg.pinv(M)
    except np.linalg.LinAlgError:
        return None
    F = float(Rb @ Mi @ Rb) / K
    if not np.isfinite(F) or F < 0:
        return None
    return {"W": K * F, "F": F, "df1": K, "df2": r.df_resid,
            "p": float(stats.f.sf(F, K, r.df_resid))}


def dumitrescu_hurlin(pnl: Panel, cause, effect, K=1):
    """Dumitrescu & Hurlin (2012) heterojen panel nedensellik testi."""
    Ws, Ts, det = [], [], []
    for u, g in pnl.df.groupby(pnl.id_var, sort=False, observed=True):
        y = g[effect].to_numpy(float)
        x = g[cause].to_numpy(float)
        r = _unit_wald(y, x, K)
        if r is None:
            continue
        Ws.append(r["W"]); Ts.append(y.size)
        det.append({"unit": str(u), "W": f(r["W"]), "F": f(r["F"]),
                    "p": f(r["p"]), "sig": bool(r["p"] < 0.05)})
    N = len(Ws)
    if N < 3:
        return None
    Wbar = float(np.mean(Ws))
    T = float(np.mean(Ts))
    Zbar = np.sqrt(N / (2.0 * K)) * (Wbar - K)
    p_zbar = float(stats.norm.sf(Zbar))
    if T > 2 * K + 5:
        Ztil = (np.sqrt(N / (2.0 * K) * (T - 2 * K - 5) / (T - K - 3)) *
                ((T - 2 * K - 3) / (T - 2 * K - 1) * Wbar - K))
        p_ztil = float(stats.norm.sf(Ztil))
    else:
        Ztil, p_ztil = np.nan, np.nan
    p_use = p_ztil if np.isfinite(p_ztil) else p_zbar
    return {
        "cause": cause, "effect": effect, "lags": K, "N": N,
        "h0": f"{cause} değişkeni {effect} değişkeninin Granger nedeni değildir",
        "Wbar": f(Wbar),
        "Zbar": f(Zbar), "pZbar": f(p_zbar),
        "Ztilde": f(Ztil), "pZtilde": f(p_ztil),
        "p": f(p_use), "star": star(p_use),
        "sig": bool(np.isfinite(p_use) and p_use < 0.05),
        "decision": "H₀ reddedilir" if (np.isfinite(p_use) and p_use < 0.05) else "H₀ reddedilemez",
        "conclusion": (f"{cause} → {effect} yönünde nedensellik vardır."
                       if (np.isfinite(p_use) and p_use < 0.05)
                       else f"{cause} → {effect} yönünde nedensellik yoktur."),
        "nSigUnits": int(sum(1 for d in det if d["sig"])),
        "units": det[:60],
    }


def pooled_granger(pnl: Panel, cause, effect, K=1):
    """Homojen (havuzlanmis, birim sabit etkili) Granger nedensellik testi."""
    Y, cols_list, gids = [], [], []
    for u, g in pnl.df.groupby(pnl.id_var, sort=False, observed=True):
        y = g[effect].to_numpy(float)
        x = g[cause].to_numpy(float)
        T = y.size
        if T < 2 * K + 4:
            continue
        idx = np.arange(K, T)
        block = []
        for k in range(1, K + 1):
            block.append(y[idx - k])
        for k in range(1, K + 1):
            block.append(x[idx - k])
        Y.append(y[idx])
        cols_list.append(np.column_stack(block))
        gids.append(np.array([u] * idx.size, dtype=object))
    if len(Y) < 2:
        return None
    Yv = np.concatenate(Y)
    Xv = np.vstack(cols_list)
    G = np.concatenate(gids)
    # birim-ici donusum (sabit etkiler)
    Yd, Xd = Yv.copy(), Xv.copy()
    for u in np.unique(G):
        m = G == u
        Yd[m] -= Yd[m].mean()
        Xd[m] -= Xd[m].mean(axis=0)
    try:
        r = ols(Yd, Xd, has_const=False)
    except Exception:
        return None
    nu = len(np.unique(G))
    dfr = max(Yv.size - Xv.shape[1] - nu, 1)
    scale = r.df_resid / dfr
    R = np.zeros((K, Xv.shape[1]))
    for k in range(K):
        R[k, K + k] = 1.0
    Rb = R @ r.beta
    M = R @ (r.cov * scale) @ R.T
    try:
        F = float(Rb @ np.linalg.pinv(M) @ Rb) / K
    except np.linalg.LinAlgError:
        return None
    p = float(stats.f.sf(F, K, dfr))
    return {"cause": cause, "effect": effect, "lags": K,
            "stat": f(F), "df1": K, "df2": int(dfr), "p": f(p),
            "star": star(p), "sig": bool(p < 0.05)}


def run_all(pnl: Panel, lags=1, max_vars=6, include_pooled=True):
    variables = pnl.vars[:max_vars]
    pairs, pooled = [], []
    for a, b in permutations(variables, 2):
        try:
            r = dumitrescu_hurlin(pnl, a, b, lags)
        except Exception:
            r = None
        if r:
            pairs.append(r)
        if include_pooled:
            try:
                pg = pooled_granger(pnl, a, b, lags)
            except Exception:
                pg = None
            if pg:
                pooled.append(pg)

    # cift yonlu / tek yonlu ozet
    summary = []
    seen = set()
    lookup = {(p["cause"], p["effect"]): p for p in pairs}
    for a, b in permutations(variables, 2):
        if (b, a) in seen:
            continue
        seen.add((a, b))
        f1, f2 = lookup.get((a, b)), lookup.get((b, a))
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
    return {"lags": lags, "pairs": pairs, "pooled": pooled, "summary": summary,
            "method": "Dumitrescu & Hurlin (2012) heterojen panel nedensellik testi"}
