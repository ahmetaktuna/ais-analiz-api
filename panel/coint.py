"""Panel esbutunlesme testleri (Pedroni, Kao, Westerlund) ve
uzun donem katsayi tahmincileri (Grup-ortalamasi FMOLS ve DOLS)."""
from __future__ import annotations


import numpy as np
from scipy import stats

from .prepare import Panel
from .utils import (add_const, f, mc_draws, newey_west_lrcov,
                    newey_west_lrv, ols, star)


def _nw_lags(T):
    return int(max(1, np.floor(4.0 * (T / 100.0) ** 0.25)))


def _demean_det(v, det, T):
    """Deterministik bilesenleri arindir. det: 'c' veya 'ct'."""
    if det == "ct":
        Z = np.column_stack([np.ones(T), np.arange(1, T + 1, dtype=float)])
    else:
        Z = np.ones((T, 1))
    b = np.linalg.pinv(Z.T @ Z) @ (Z.T @ v)
    return v - Z @ b


# ==========================================================================
# PEDRONI
# ==========================================================================

def _pedroni_raw(units, det="c", lags=1):
    """
    units: [(y_i (T,), X_i (T,k)), ...]
    Pedroni'nin 7 ham istatistigini dondurur.
    """
    num_v = 0.0
    sum_e2lag_L = 0.0          # sum L^-2 e_{t-1}^2
    sum_num_rho = 0.0          # sum L^-2 (e_{t-1} de_t - lambda)
    grp_rho, grp_pp, grp_adf = [], [], []
    sig2_list = []
    sum_e2lag_star = 0.0
    sum_num_adf = 0.0
    used = 0

    for y, X in units:
        T = y.size
        k = X.shape[1]
        if T < max(8, 3 * lags + 6):
            continue
        # esbutunlesme regresyonu
        det_cols = [np.ones(T)]
        if det == "ct":
            det_cols.append(np.arange(1, T + 1, dtype=float))
        Z = np.column_stack(det_cols + [X])
        try:
            b = np.linalg.pinv(Z.T @ Z) @ (Z.T @ y)
        except np.linalg.LinAlgError:
            continue
        e = y - Z @ b
        if not np.all(np.isfinite(e)):
            continue

        # L11: Dy ~ Dx artiklarinin uzun donem std sapmasi
        dy = np.diff(y)
        dX = np.diff(X, axis=0)
        try:
            eta = dy - dX @ (np.linalg.pinv(dX.T @ dX) @ (dX.T @ dy))
        except np.linalg.LinAlgError:
            eta = dy - dy.mean()
        L2 = newey_west_lrv(eta, _nw_lags(T))
        if not np.isfinite(L2) or L2 <= 1e-12:
            continue

        elag, de = e[:-1], np.diff(e)
        den_i = float(elag @ elag)
        if den_i <= 0:
            continue
        # AR(1) artiklarindan lambda ve sigma
        gam = float(elag @ e[1:]) / den_i
        u = e[1:] - gam * elag
        n_u = u.size
        s2i = float(u @ u) / n_u
        m = _nw_lags(T)
        lam = 0.0
        for s in range(1, m + 1):
            w = 1.0 - s / (m + 1.0)
            lam += w * float(u[s:] @ u[:-s]) / n_u
        sig2i = s2i + 2.0 * lam
        if sig2i <= 0:
            continue
        sig2_list.append(sig2i)

        num_i = float(elag @ de) - n_u * lam
        sum_e2lag_L += den_i / L2
        sum_num_rho += num_i / L2

        grp_rho.append(T * num_i / den_i)
        grp_pp.append(num_i / np.sqrt(sig2i * den_i))

        # ADF (parametrik) versiyon
        if lags > 0 and de.size > lags + 3:
            Y = de[lags:]
            cols = [elag[lags:]]
            for j in range(1, lags + 1):
                cols.append(de[lags - j: de.size - j])
            W = np.column_stack(cols)
        else:
            Y, W = de, elag.reshape(-1, 1)
        if W.shape[0] <= W.shape[1] + 1:
            continue
        try:
            Wi = np.linalg.pinv(W.T @ W)
            bw = Wi @ (W.T @ Y)
        except np.linalg.LinAlgError:
            continue
        uw = Y - W @ bw
        dofw = W.shape[0] - W.shape[1]
        s2star = float(uw @ uw) / dofw
        elag_s = W[:, 0]
        den_s = float(elag_s @ elag_s)
        if den_s <= 0 or s2star <= 0:
            continue
        sum_e2lag_star += den_s / L2
        sum_num_adf += float(elag_s @ Y) / L2
        grp_adf.append(float(elag_s @ Y) / np.sqrt(s2star * den_s))
        num_v += 1
        used += 1

    if used < 2 or sum_e2lag_L <= 0 or sum_e2lag_star <= 0:
        return None

    N = used
    s2_bar = float(np.mean(sig2_list))
    out = {
        "panel_v": (used ** 1.5) * (np.mean([u[0].size for u in units]) ** 2) / sum_e2lag_L
        if sum_e2lag_L > 0 else np.nan,
        "panel_rho": sum_num_rho / sum_e2lag_L * np.sqrt(N) *
        float(np.mean([u[0].size for u in units])),
        "panel_pp": sum_num_rho / np.sqrt(s2_bar * sum_e2lag_L),
        "panel_adf": sum_num_adf / np.sqrt(s2_bar * sum_e2lag_star),
        "group_rho": float(np.sum(grp_rho)) / np.sqrt(N) if grp_rho else np.nan,
        "group_pp": float(np.sum(grp_pp)) / np.sqrt(N) if grp_pp else np.nan,
        "group_adf": float(np.sum(grp_adf)) / np.sqrt(N) if grp_adf else np.nan,
        "_N": N,
    }
    return out


PEDRONI_META = [
    ("panel_v", "Panel v-İstatistiği", "right", "İçsel (within) boyut"),
    ("panel_rho", "Panel ρ-İstatistiği", "left", "İçsel (within) boyut"),
    ("panel_pp", "Panel PP-İstatistiği", "left", "İçsel (within) boyut"),
    ("panel_adf", "Panel ADF-İstatistiği", "left", "İçsel (within) boyut"),
    ("group_rho", "Grup ρ-İstatistiği", "left", "Gruplar arası (between) boyut"),
    ("group_pp", "Grup PP-İstatistiği", "left", "Gruplar arası (between) boyut"),
    ("group_adf", "Grup ADF-İstatistiği", "left", "Gruplar arası (between) boyut"),
]


def _pedroni_null(N: int, T: int, k: int, det: str, lags: int, reps: int,
                  seed: int = 90210):
    def draw(rng):
        units = []
        for _i in range(N):
            y = np.cumsum(rng.standard_normal(T))
            X = np.cumsum(rng.standard_normal((T, k)), axis=0)
            units.append((y, X))
        return _pedroni_raw(units, det, lags) or {}

    rows, _ = mc_draws(("pedroni", N, T, k, det, lags), reps,
                       seed + 13 * N + 7 * T + 3 * k + lags + len(det), draw)
    acc = {key: [] for key, *_ in PEDRONI_META}
    for r in rows:
        for key, *_ in PEDRONI_META:
            v = r.get(key)
            if v is not None and np.isfinite(v):
                acc[key].append(v)
    return {kk: np.asarray(vv) for kk, vv in acc.items()}


def pedroni(pnl: Panel, det="c", lags=1, reps=200):
    units = []
    for u, g in pnl.df.groupby(pnl.id_var, sort=False, observed=True):
        y = g[pnl.dep].to_numpy(float)
        X = g[pnl.indep].to_numpy(float)
        if y.size >= 8:
            units.append((y, X))
    if len(units) < 3:
        return None
    obs = _pedroni_raw(units, det, lags)
    if not obs:
        return None
    N = obs["_N"]
    T = int(np.median([u[0].size for u in units]))
    k = len(pnl.indep)
    null = _pedroni_null(N, T, k, det, int(lags), int(reps))

    rows = []
    nsig = 0
    for key, label, tail, dim in PEDRONI_META:
        val = obs.get(key)
        nl = null.get(key, np.asarray([]))
        if val is None or not np.isfinite(val) or nl.size < 30:
            rows.append({"stat": label, "dimension": dim, "value": f(val),
                         "z": None, "p": None, "sig": False})
            continue
        mu, sd = float(nl.mean()), float(nl.std(ddof=1))
        z = (val - mu) / sd if sd > 0 else np.nan
        if tail == "right":
            p = float((nl >= val).mean())
        else:
            p = float((nl <= val).mean())
        sig = bool(p < 0.05)
        nsig += int(sig)
        rows.append({"stat": label, "dimension": dim, "value": f(val),
                     "z": f(z), "p": f(p), "sig": sig, "star": star(p)})
    return {
        "name": "Pedroni (1999, 2004) Panel Eşbütünleşme Testi",
        "h0": "Eşbütünleşme ilişkisi yoktur",
        "det": det, "lags": lags, "N": N, "T": T,
        "reps": int(min((len(v) for v in null.values() if len(v)), default=reps)),
        "repsRequested": int(reps),
        "rows": rows, "nSignificant": nsig, "nTests": len(PEDRONI_META),
        "decision": ("Eşbütünleşme vardır" if nsig >= 4 else
                     "Kısmi kanıt" if nsig >= 2 else "Eşbütünleşme yoktur"),
        "note": "p-değerleri, aynı N, T ve regresör sayısı için sıfır hipotezi "
                "altında Monte Carlo simülasyonu ile üretilmiştir.",
    }


# ==========================================================================
# KAO
# ==========================================================================

def kao(pnl: Panel, lags=1):
    """Kao (1999) artik tabanli ADF esbutunlesme testi."""
    d = pnl.df
    # birim-ici (within) donusum
    y = d[pnl.dep].to_numpy(float)
    X = d[pnl.indep].to_numpy(float)
    gid = d[pnl.id_var].to_numpy()
    ys, Xs = y.copy(), X.copy()
    for u in np.unique(gid):
        m = gid == u
        ys[m] -= ys[m].mean()
        Xs[m] -= Xs[m].mean(axis=0)
    try:
        b = np.linalg.pinv(Xs.T @ Xs) @ (Xs.T @ ys)
    except np.linalg.LinAlgError:
        return None
    e = ys - Xs @ b

    # havuzlanmis ADF regresyonu (artiklar uzerinde)
    Y, L, D = [], [], []
    dx_all, u_all = [], []
    for u in np.unique(gid):
        m = gid == u
        ei = e[m]
        if ei.size < lags + 4:
            continue
        de = np.diff(ei)
        elag = ei[:-1]
        if lags > 0 and de.size > lags + 2:
            Y.append(de[lags:])
            L.append(elag[lags:])
            Dj = [de[lags - j: de.size - j] for j in range(1, lags + 1)]
            D.append(np.column_stack(Dj))
        else:
            Y.append(de); L.append(elag)
            D.append(np.zeros((de.size, 0)))
        dx_all.append(np.diff(X[m], axis=0))
        u_all.append(ei[1:])
    if len(Y) < 2:
        return None
    Yv = np.concatenate(Y)
    Lv = np.concatenate(L)
    Dv = np.vstack(D) if D[0].shape[1] > 0 else np.zeros((Yv.size, 0))
    W = np.column_stack([Lv, Dv]) if Dv.shape[1] else Lv.reshape(-1, 1)
    try:
        r = ols(Yv, W, has_const=False)
    except Exception:
        return None
    t_adf = float(r.tstat[0])

    # uzun donem kovaryanslar
    n_l = min(len(u_all), len(dx_all))
    U = np.concatenate([u_all[i] for i in range(n_l)])
    DX = np.vstack([dx_all[i] for i in range(n_l)])
    n = min(U.size, DX.shape[0])
    M = np.column_stack([U[:n], DX[:n]])
    S = M.T @ M / n
    Om = newey_west_lrcov(M, _nw_lags(n))
    k = DX.shape[1]
    s_uu, s_ux, s_xx = S[0, 0], S[0, 1:], S[1:, 1:]
    o_uu, o_ux, o_xx = Om[0, 0], Om[0, 1:], Om[1:, 1:]
    try:
        sv2 = float(s_uu - s_ux @ np.linalg.pinv(s_xx) @ s_ux)
        s0v2 = float(o_uu - o_ux @ np.linalg.pinv(o_xx) @ o_ux)
    except np.linalg.LinAlgError:
        return None
    if sv2 <= 0 or s0v2 <= 0:
        return None
    N = len(Y)
    adf_star = ((t_adf + np.sqrt(6.0 * N) * np.sqrt(sv2) / (2.0 * np.sqrt(s0v2))) /
                np.sqrt(s0v2 / (2.0 * sv2) + 3.0 * sv2 / (10.0 * s0v2)))
    p = float(stats.norm.cdf(adf_star))
    return {"name": "Kao (1999) Artık Tabanlı Eşbütünleşme Testi",
            "h0": "Eşbütünleşme ilişkisi yoktur",
            "tADF": f(t_adf), "stat": f(adf_star), "p": f(p),
            "dist": "N(0,1)", "lags": lags, "N": N,
            "sig": bool(p < 0.05),
            "decision": "Eşbütünleşme vardır" if p < 0.05 else "Eşbütünleşme yoktur"}


# ==========================================================================
# WESTERLUND (ECM tabanli)
# ==========================================================================

def _westerlund_raw(units, det="c", p=1, q=1):
    gts, gas, num_p, den_p, sse, nobs_p = [], [], 0.0, 0.0, 0.0, 0
    rows_y, rows_Z = [], []
    for y, X in units:
        T = y.size
        k = X.shape[1]
        start = max(p, q) + 1
        if T - start < k + p + q + 4:
            continue
        dy = np.diff(y)
        dX = np.diff(X, axis=0)
        idx = np.arange(start - 1, dy.size)
        if idx.size < k + p + q + 3:
            continue
        Yv = dy[idx]
        cols = [np.ones(idx.size)]
        if det == "ct":
            cols.append(np.arange(1, idx.size + 1, dtype=float))
        cols.append(y[idx])                       # y_{t-1}
        for j in range(k):
            cols.append(X[idx, j])                # x_{t-1}
        for j in range(1, p + 1):
            cols.append(dy[idx - j])
        for j in range(0, q + 1):
            for m2 in range(k):
                cols.append(dX[idx - j, m2])
        Z = np.column_stack(cols)
        if Z.shape[0] <= Z.shape[1] + 2:
            continue
        try:
            rr = ols(Yv, Z, has_const=True)
        except Exception:
            continue
        a_idx = 2 if det == "ct" else 1
        alpha = float(rr.beta[a_idx])
        se_a = float(rr.se[a_idx])
        if se_a <= 0 or not np.isfinite(alpha):
            continue
        gts.append(alpha / se_a)
        lag_start = a_idx + 1 + k
        sum_ay = float(np.sum(rr.beta[lag_start: lag_start + p])) if p > 0 else 0.0
        denom = 1.0 - sum_ay
        if abs(denom) < 1e-8:
            continue
        gas.append(T * alpha / denom)
        rows_y.append(Yv)
        rows_Z.append(Z)
    if len(gts) < 3:
        return None

    # havuzlanmis (panel) tahmin: ortak alpha, birime ozgu diger katsayilar
    # kismi arindirma ile
    Yp, Vp = [], []
    for Yv, Z in zip(rows_y, rows_Z):
        a_idx = 2 if det == "ct" else 1
        v = Z[:, a_idx].copy()
        Zo = np.delete(Z, a_idx, axis=1)
        try:
            P = Zo @ (np.linalg.pinv(Zo.T @ Zo) @ Zo.T)
        except np.linalg.LinAlgError:
            continue
        Yp.append(Yv - P @ Yv)
        Vp.append(v - P @ v)
    if len(Yp) < 3:
        return None
    Ya = np.concatenate(Yp)
    Va = np.concatenate(Vp)
    den = float(Va @ Va)
    if den <= 0:
        return None
    alpha_p = float(Va @ Ya) / den
    res = Ya - alpha_p * Va
    s2 = float(res @ res) / max(Ya.size - 1, 1)
    se_p = np.sqrt(s2 / den)
    Tbar = float(np.mean([u[0].size for u in units]))
    return {
        "Gt": float(np.mean(gts)),
        "Ga": float(np.mean(gas)),
        "Pt": float(alpha_p / se_p) if se_p > 0 else np.nan,
        "Pa": float(Tbar * alpha_p),
        "_N": len(gts),
    }


WEST_META = [("Gt", "Gt", "Grup ortalaması"), ("Ga", "Ga", "Grup ortalaması"),
             ("Pt", "Pt", "Panel"), ("Pa", "Pa", "Panel")]


def _westerlund_null(N: int, T: int, k: int, det: str, p: int, q: int,
                     reps: int, seed: int = 314159):
    def draw(rng):
        units = [(np.cumsum(rng.standard_normal(T)),
                  np.cumsum(rng.standard_normal((T, k)), axis=0)) for _ in range(N)]
        return _westerlund_raw(units, det, p, q) or {}

    rows, _ = mc_draws(("west", N, T, k, det, p, q), reps,
                       seed + 17 * N + 5 * T + k + p + q + len(det), draw)
    acc = {key: [] for key, *_ in WEST_META}
    for r in rows:
        for key, *_ in WEST_META:
            v = r.get(key)
            if v is not None and np.isfinite(v):
                acc[key].append(v)
    return {kk: np.asarray(vv) for kk, vv in acc.items()}


def westerlund(pnl: Panel, det="c", p=1, q=1, reps=200):
    units = []
    for u, g in pnl.df.groupby(pnl.id_var, sort=False, observed=True):
        y = g[pnl.dep].to_numpy(float)
        X = g[pnl.indep].to_numpy(float)
        if y.size >= max(10, 2 * (p + q) + 8):
            units.append((y, X))
    if len(units) < 3:
        return None
    obs = _westerlund_raw(units, det, p, q)
    if not obs:
        return None
    N = obs["_N"]
    T = int(np.median([u[0].size for u in units]))
    k = len(pnl.indep)
    null = _westerlund_null(N, T, k, det, int(p), int(q), int(reps))
    rows, nsig = [], 0
    for key, label, dim in WEST_META:
        val = obs.get(key)
        nl = null.get(key, np.asarray([]))
        if val is None or not np.isfinite(val) or nl.size < 30:
            rows.append({"stat": label, "dimension": dim, "value": f(val),
                         "z": None, "p": None, "sig": False})
            continue
        mu, sd = float(nl.mean()), float(nl.std(ddof=1))
        z = (val - mu) / sd if sd > 0 else np.nan
        pv = float((nl <= val).mean())
        sig = bool(pv < 0.05)
        nsig += int(sig)
        rows.append({"stat": label, "dimension": dim, "value": f(val),
                     "z": f(z), "p": f(pv), "sig": sig, "star": star(pv)})
    return {"name": "Westerlund (2007) Hata Düzeltme Tabanlı Panel Eşbütünleşme Testi",
            "h0": "Eşbütünleşme ilişkisi yoktur (hata düzeltme terimi anlamsızdır)",
            "rows": rows, "N": N, "T": T,
            "reps": int(min((len(v) for v in null.values() if len(v)), default=reps)),
            "repsRequested": int(reps),
            "nSignificant": nsig, "nTests": len(WEST_META),
            "decision": ("Eşbütünleşme vardır" if nsig >= 2 else
                         "Eşbütünleşme yoktur"),
            "note": "p-değerleri Monte Carlo simülasyonu ile üretilmiştir."}


# ==========================================================================
# Uzun donem katsayilar: Grup-ortalamasi FMOLS ve DOLS
# ==========================================================================

def group_fmols(pnl: Panel):
    k = len(pnl.indep)
    betas, tvals = [], []
    for u, g in pnl.df.groupby(pnl.id_var, sort=False, observed=True):
        y = g[pnl.dep].to_numpy(float)
        X = g[pnl.indep].to_numpy(float)
        T = y.size
        if T < max(10, 3 * k + 6):
            continue
        yt = y - y.mean()
        Xt = X - X.mean(axis=0)
        try:
            b0 = np.linalg.pinv(Xt.T @ Xt) @ (Xt.T @ yt)
        except np.linalg.LinAlgError:
            continue
        u_hat = yt - Xt @ b0
        dX = np.diff(X, axis=0)
        dX = dX - dX.mean(axis=0)
        n = min(u_hat.size - 1, dX.shape[0])
        Xi = np.column_stack([u_hat[1:1 + n], dX[:n]])
        m = _nw_lags(n)
        Om = newey_west_lrcov(Xi, m)
        S0 = Xi.T @ Xi / n
        # tek yonlu toplam Gamma
        Gam = np.zeros_like(S0)
        for s in range(1, m + 1):
            w = 1.0 - s / (m + 1.0)
            Gam += w * (Xi[s:].T @ Xi[:-s] / n)
        O11, O21, O22 = Om[0, 0], Om[1:, 0], Om[1:, 1:]
        G21, G22 = Gam[1:, 0], Gam[1:, 1:]
        S21, S22 = S0[1:, 0], S0[1:, 1:]
        try:
            O22i = np.linalg.pinv(O22)
        except np.linalg.LinAlgError:
            continue
        adj = O22i @ O21                      # (k,)
        y_star = yt[1:1 + n] - dX[:n] @ adj
        gamma = (G21 + S21) - (G22 + S22) @ adj
        Xs = Xt[1:1 + n]
        try:
            M = np.linalg.pinv(Xs.T @ Xs)
        except np.linalg.LinAlgError:
            continue
        b_i = M @ (Xs.T @ y_star - n * gamma)
        o112 = float(O11 - O21 @ adj)
        if o112 <= 0:
            continue
        se_i = np.sqrt(np.maximum(np.diag(o112 * M), 1e-300))
        betas.append(b_i)
        tvals.append(b_i / se_i)
    if len(betas) < 2:
        return None
    B = np.vstack(betas)
    Tv = np.vstack(tvals)
    bbar = B.mean(axis=0)
    N = B.shape[0]
    tbar = Tv.sum(axis=0) / np.sqrt(N)
    se_group = B.std(axis=0, ddof=1) / np.sqrt(N)
    pv = 2 * stats.norm.sf(np.abs(tbar))
    return {"name": "Grup Ortalaması FMOLS (Pedroni, 2001)",
            "N": N,
            "coeffs": [{"name": v, "B": f(bbar[j]), "SE": f(se_group[j]),
                        "t": f(tbar[j]), "p": f(pv[j]), "sig": star(pv[j])}
                       for j, v in enumerate(pnl.indep)]}


def group_dols(pnl: Panel, leads=1, lags=1):
    k = len(pnl.indep)
    betas, tvals = [], []
    for u, g in pnl.df.groupby(pnl.id_var, sort=False, observed=True):
        y = g[pnl.dep].to_numpy(float)
        X = g[pnl.indep].to_numpy(float)
        T = y.size
        need = k * (leads + lags + 1) + 4
        if T < need + 6:
            continue
        dX = np.vstack([np.zeros((1, k)), np.diff(X, axis=0)])
        lo, hi = lags, T - leads
        if hi - lo < need:
            continue
        idx = np.arange(lo, hi)
        cols = [np.ones(idx.size), *[X[idx, j] for j in range(k)]]
        for s in range(-leads, lags + 1):
            if s == 0:
                for j in range(k):
                    cols.append(dX[idx, j])
            else:
                for j in range(k):
                    cols.append(dX[idx + s, j])
        Z = np.column_stack(cols)
        if Z.shape[0] <= Z.shape[1] + 2:
            continue
        try:
            r = ols(y[idx], Z, has_const=True)
        except Exception:
            continue
        b_i = r.beta[1:1 + k]
        # HAC standart hata
        e = r.resid
        m = _nw_lags(e.size)
        Om = newey_west_lrcov(Z * e[:, None], m) * Z.shape[0]
        V = r.XtXi @ Om @ r.XtXi
        se_i = np.sqrt(np.maximum(np.diag(V)[1:1 + k], 1e-300))
        betas.append(b_i)
        tvals.append(b_i / se_i)
    if len(betas) < 2:
        return None
    B = np.vstack(betas)
    Tv = np.vstack(tvals)
    N = B.shape[0]
    bbar = B.mean(axis=0)
    tbar = Tv.sum(axis=0) / np.sqrt(N)
    se_group = B.std(axis=0, ddof=1) / np.sqrt(N)
    pv = 2 * stats.norm.sf(np.abs(tbar))
    return {"name": f"Grup Ortalaması DOLS (öncül={leads}, gecikme={lags})",
            "N": N,
            "coeffs": [{"name": v, "B": f(bbar[j]), "SE": f(se_group[j]),
                        "t": f(tbar[j]), "p": f(pv[j]), "sig": star(pv[j])}
                       for j, v in enumerate(pnl.indep)]}


# ==========================================================================

def run_all(pnl: Panel, det="c", lags=1, reps=200, longrun=True):
    out = {"pedroni": None, "kao": None, "westerlund": None,
           "fmols": None, "dols": None, "options": {
               "det": det, "lags": lags, "mcReps": reps}}
    try:
        out["pedroni"] = pedroni(pnl, det, lags, reps)
    except Exception as exc:
        out["pedroniError"] = str(exc)
    try:
        out["kao"] = kao(pnl, lags)
    except Exception as exc:
        out["kaoError"] = str(exc)
    try:
        out["westerlund"] = westerlund(pnl, det, max(lags, 1), max(lags, 1), reps)
    except Exception as exc:
        out["westerlundError"] = str(exc)
    if longrun:
        try:
            out["fmols"] = group_fmols(pnl)
        except Exception as exc:
            out["fmolsError"] = str(exc)
        try:
            out["dols"] = group_dols(pnl)
        except Exception as exc:
            out["dolsError"] = str(exc)

    votes = []
    if out["pedroni"]:
        votes.append(out["pedroni"]["nSignificant"] >= 4)
    if out["kao"]:
        votes.append(bool(out["kao"]["sig"]))
    if out["westerlund"]:
        votes.append(out["westerlund"]["nSignificant"] >= 2)
    if votes:
        yes = sum(votes)
        out["overall"] = {
            "yes": yes, "total": len(votes),
            "cointegrated": yes >= (len(votes) / 2.0),
            "text": (f"{len(votes)} test grubundan {yes} tanesi eşbütünleşme "
                     "bulgusunu desteklemektedir."),
        }
    return out
