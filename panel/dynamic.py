"""Dinamik panel: Arellano-Bond / Blundell-Bond GMM ve Panel ARDL (MG / PMG / DFE)."""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from .prepare import Panel
from .utils import f, ols, star


# ==========================================================================
# GMM ALTYAPISI
# ==========================================================================

def _H_matrix(m):
    H = np.zeros((m, m))
    np.fill_diagonal(H, 2.0)
    for i in range(m - 1):
        H[i, i + 1] = -1.0
        H[i + 1, i] = -1.0
    return H


def _build_diff_gmm(pnl: Panel, ylag=1, max_inst=4, time_dummies=False,
                    collapse=False):
    """Fark GMM icin (dY, dX, Z) bloklarini olusturur.

    collapse=True  -> Roodman (2009) sadelestirilmis arac matrisi
                      (her gecikme derinligi icin tek sutun).
    collapse=False -> standart (her donem x gecikme derinligi icin ayri sutun).
    """
    times = sorted(pnl.df[pnl.time_var].unique())
    tindex = {t: i for i, t in enumerate(times)}
    blocks = []
    inst_slots = []
    for u, g in pnl.df.groupby(pnl.id_var, sort=False, observed=True):
        y = g[pnl.dep].to_numpy(float)
        X = g[pnl.indep].to_numpy(float)
        tt = [tindex[t] for t in g[pnl.time_var]]
        T = y.size
        if T < ylag + 3:
            continue
        rows = []
        for s in range(ylag + 1, T):
            dy = y[s] - y[s - 1]
            dylags = [y[s - j] - y[s - j - 1] for j in range(1, ylag + 1)]
            dx = X[s] - X[s - 1]
            # gecikme derinligi d: y_{t-d}, d = ylag+1, ylag+2, ...
            depths = list(range(ylag + 1, s + 1))
            if max_inst:
                depths = depths[:max_inst]
            inst = [((0 if collapse else tt[s]), d, y[s - d]) for d in depths]
            rows.append({"t": tt[s], "dy": dy, "dylags": dylags, "dx": dx,
                         "inst": inst})
        if rows:
            blocks.append((str(u), rows))
            inst_slots.extend([(a, b) for r in rows for a, b, _ in r["inst"]])
    if not blocks:
        return None
    keys = sorted(set(inst_slots))
    kmap = {k: i for i, k in enumerate(keys)}
    n_inst_y = len(keys)
    kx = len(pnl.indep)
    td = sorted({r["t"] for _, rows in blocks for r in rows}) if time_dummies else []
    tdmap = {t: i for i, t in enumerate(td[1:])} if time_dummies else {}
    n_z = n_inst_y + kx + len(tdmap)

    out = []
    for u, rows in blocks:
        m = len(rows)
        dY = np.zeros(m)
        dXm = np.zeros((m, ylag + kx + len(tdmap)))
        Z = np.zeros((m, n_z))
        for i, r in enumerate(rows):
            dY[i] = r["dy"]
            dXm[i, :ylag] = r["dylags"]
            dXm[i, ylag:ylag + kx] = r["dx"]
            Z[i, n_inst_y:n_inst_y + kx] = r["dx"]
            for a, b_, val in r["inst"]:
                Z[i, kmap[(a, b_)]] += val
            if time_dummies and r["t"] in tdmap:
                col = tdmap[r["t"]]
                dXm[i, ylag + kx + col] = 1.0
                Z[i, n_inst_y + kx + col] = 1.0
        out.append((u, dY, dXm, Z))
    names = ([f"{pnl.dep} (t−{j})" for j in range(1, ylag + 1)] +
             list(pnl.indep) + [f"D_{t}" for t in tdmap])
    return out, names, n_z


def _gmm_estimate(blocks, W):
    XZ = 0.0
    Zy = 0.0
    for _, dY, dX, Z in blocks:
        XZ = XZ + dX.T @ Z
        Zy = Zy + Z.T @ dY
    A = XZ @ W @ XZ.T
    b = np.linalg.pinv(A) @ (XZ @ W @ Zy)
    return b, XZ, Zy, A


def _residuals(blocks, b):
    return [(u, dY - dX @ b, dX, Z) for u, dY, dX, Z in blocks]


def arellano_bond(pnl: Panel, ylag=1, max_inst=3, twostep=True,
                  time_dummies=False, collapse=True):
    """Arellano-Bond (1991) fark GMM tahmincisi."""
    built = _build_diff_gmm(pnl, ylag, max_inst, time_dummies, collapse)
    if not built:
        return None
    blocks, names, n_z = built
    k = blocks[0][2].shape[1]
    if n_z < k:
        return None
    N = len(blocks)

    # --- bir asamali
    W1 = 0.0
    for _, dY, dX, Z in blocks:
        m = Z.shape[0]
        W1 = W1 + Z.T @ _H_matrix(m) @ Z
    W1 = np.linalg.pinv(W1)
    b1, XZ, Zy, A1 = _gmm_estimate(blocks, W1)
    res1 = _residuals(blocks, b1)

    Om1 = 0.0
    for _, e, dX, Z in res1:
        s = Z.T @ e
        Om1 = Om1 + np.outer(s, s)

    A1i = np.linalg.pinv(A1)
    V1r = A1i @ (XZ @ W1 @ Om1 @ W1 @ XZ.T) @ A1i     # bir asamali dirençli

    result = {"names": names, "N": N, "nInstruments": int(n_z),
              "nParams": int(k), "ylag": ylag, "maxInstruments": max_inst,
              "collapse": bool(collapse), "timeDummies": bool(time_dummies)}

    if not twostep:
        b, V = b1, V1r
        step = "Tek aşamalı (dirençli)"
        step_short = "tek aşamalı ve dirençli standart hatalı"
    else:
        W2 = np.linalg.pinv(Om1)
        b2, XZ2, Zy2, A2 = _gmm_estimate(blocks, W2)
        A2i = np.linalg.pinv(A2)
        V2 = A2i                                  # standart iki asamali
        res2 = _residuals(blocks, b2)
        # Windmeijer (2005) sonlu ornek duzeltmesi
        D = np.zeros((k, k))
        Zv2 = sum(Z.T @ e for _, e, _, Z in res2)
        for kk in range(k):
            dOm = 0.0
            for (_, e1, dX, Z) in res1:
                xk = dX[:, kk]
                s1 = Z.T @ e1
                s2 = Z.T @ xk
                dOm = dOm - (np.outer(s1, s2) + np.outer(s2, s1))
            D[:, kk] = -(A2i @ (XZ2 @ W2 @ dOm @ W2 @ Zv2))
        Vc = V2 + D @ V2 + (D @ V2).T + D @ V1r @ D.T
        b, V = b2, Vc
        step = "İki aşamalı (Windmeijer düzeltmeli)"
        step_short = "iki aşamalı ve Windmeijer sonlu örnek düzeltmeli"

    se = np.sqrt(np.maximum(np.diag(V), 1e-300))
    z = b / se
    p = 2 * stats.norm.sf(np.abs(z))
    result["step"] = step
    result["stepShort"] = step_short
    result["coeffs"] = [{"name": names[j], "B": f(b[j]), "SE": f(se[j]),
                         "z": f(z[j]), "t": f(z[j]), "p": f(p[j]),
                         "sig": star(p[j]),
                         "ciLow": f(b[j] - 1.96 * se[j]),
                         "ciHigh": f(b[j] + 1.96 * se[j])} for j in range(k)]

    # kalicilik / uzun donem katsayilari
    rho = float(np.sum(b[:ylag]))
    result["persistence"] = f(rho)
    if abs(1 - rho) > 1e-6:
        kx = len(pnl.indep)
        result["longRun"] = [{"name": pnl.indep[j], "B": f(b[ylag + j] / (1 - rho))}
                             for j in range(kx)]
        result["stable"] = bool(abs(rho) < 1)

    # --- Sargan / Hansen
    resf = _residuals(blocks, b)
    Zv = sum(Z.T @ e for _, e, _, Z in resf)
    sargan = float(Zv @ W1 @ Zv)
    dfj = int(n_z - k)
    result["sargan"] = {"name": "Sargan Aşırı Tanımlama Testi",
                        "h0": "Araç değişkenler geçerlidir",
                        "stat": f(sargan), "df": dfj,
                        "p": f(float(stats.chi2.sf(sargan, dfj))) if dfj > 0 else None}
    if twostep:
        hansen = float(Zv @ np.linalg.pinv(Om1) @ Zv)
        result["hansen"] = {"name": "Hansen J Testi",
                            "h0": "Araç değişkenler geçerlidir",
                            "stat": f(hansen), "df": dfj,
                            "p": f(float(stats.chi2.sf(hansen, dfj))) if dfj > 0 else None}

    # --- Arellano-Bond AR(1) / AR(2) testleri (birim bazinda kumelenmis)
    result["ar"] = []
    for order in (1, 2):
        contrib = [float(e[order:] @ e[:-order]) for _, e, _, _ in resf
                   if e.size > order]
        if len(contrib) < 4:
            continue
        c = np.asarray(contrib, float)
        v = float(c @ c)
        if v <= 0:
            continue
        zst = float(c.sum() / np.sqrt(v))
        pv = float(2 * stats.norm.sf(abs(zst)))
        result["ar"].append({
            "order": order, "stat": f(zst), "p": f(pv),
            "h0": f"Fark alınmış hatalarda {order}. derece otokorelasyon yoktur",
            "expected": ("Reddedilmesi beklenir (normal)" if order == 1
                         else "Reddedilmemelidir"),
            "ok": bool(pv > 0.05) if order == 2 else True,
        })

    # --- arac degiskenlerin cokluguna dair uyarilar
    warn = []
    if n_z > N:
        warn.append(f"Araç değişken sayısı ({n_z}) birim sayısından ({N}) fazladır; "
                    "Hansen J testi aşırı uyum nedeniyle güvenilirliğini yitirebilir. "
                    "Araç değişken gecikme sayısını azaltmayı veya araçları "
                    "sadeleştirmeyi (collapse) deneyiniz.")
    hp = (result.get("hansen") or {}).get("p")
    if hp is not None and hp > 0.25 and n_z > N:
        warn.append(f"Hansen J testi p-değeri ({hp}) beklenenden yüksektir; bu durum "
                    "araç değişken enflasyonunun bir göstergesidir.")
    result["warnings"] = warn
    result["instrumentWarning"] = bool(n_z > N)
    return result


# ==========================================================================
# PANEL ARDL: MG / PMG / DFE
# ==========================================================================

def _ecm_blocks(pnl: Panel, p=1, q=1):
    """Her birim icin ECM regresyon bloklarini kurar."""
    k = len(pnl.indep)
    out = []
    for u, g in pnl.df.groupby(pnl.id_var, sort=False, observed=True):
        y = g[pnl.dep].to_numpy(float)
        X = g[pnl.indep].to_numpy(float)
        T = y.size
        start = max(p, q) + 1
        need = 2 + k + (p - 1 if p > 0 else 0) + k * q + 4
        if T - start < need:
            continue
        dy = np.diff(y)
        dX = np.diff(X, axis=0)
        idx = np.arange(start - 1, dy.size)
        if idx.size < need:
            continue
        Yv = dy[idx]
        ylag = y[idx]                 # y_{t-1}
        xlag = X[idx, :]              # x_{t-1}
        short = [np.ones(idx.size)]
        for j in range(1, p):
            short.append(dy[idx - j])
        for j in range(0, q):
            for m in range(k):
                short.append(dX[idx - j, m])
        S = np.column_stack(short)
        out.append({"unit": str(u), "dy": Yv, "ylag": ylag, "xlag": xlag, "S": S})
    return out


def mean_group(pnl: Panel, p=1, q=1):
    blocks = _ecm_blocks(pnl, p, q)
    k = len(pnl.indep)
    thetas, phis, ok = [], [], []
    for b in blocks:
        Z = np.column_stack([b["S"], b["ylag"].reshape(-1, 1), b["xlag"]])
        if Z.shape[0] <= Z.shape[1] + 2:
            continue
        try:
            r = ols(b["dy"], Z, has_const=True)
        except Exception:
            continue
        ns = b["S"].shape[1]
        phi = float(r.beta[ns])
        beta = r.beta[ns + 1: ns + 1 + k]
        if abs(phi) < 1e-8 or not np.isfinite(phi):
            continue
        thetas.append(-beta / phi)
        phis.append(phi)
        ok.append(b["unit"])
    N = len(thetas)
    if N < 3:
        return None
    Th = np.vstack(thetas)
    tbar = Th.mean(axis=0)
    se = Th.std(axis=0, ddof=1) / np.sqrt(N)
    t = np.divide(tbar, se, out=np.full_like(tbar, np.nan), where=se > 0)
    pv = 2 * stats.norm.sf(np.abs(t))
    ph = np.asarray(phis)
    pbar = float(ph.mean())
    pse = float(ph.std(ddof=1) / np.sqrt(N))
    pt = pbar / pse if pse > 0 else np.nan
    return {
        "name": "Ortalama Grup (Mean Group, MG) Tahmincisi",
        "N": N, "p": p, "q": q,
        "longRun": [{"name": v, "B": f(tbar[j]), "SE": f(se[j]), "t": f(t[j]),
                     "p": f(pv[j]), "sig": star(pv[j])} for j, v in enumerate(pnl.indep)],
        "ecm": {"name": "Hata Düzeltme Katsayısı (φ)", "B": f(pbar), "SE": f(pse),
                "t": f(pt), "p": f(float(2 * stats.norm.sf(abs(pt)))) if np.isfinite(pt) else None},
        "theta": tbar.tolist(), "thetaVar": (Th.var(axis=0, ddof=1) / N).tolist(),
        "units": ok[:60],
    }


def dynamic_fe(pnl: Panel, p=1, q=1):
    blocks = _ecm_blocks(pnl, p, q)
    k = len(pnl.indep)
    if len(blocks) < 3:
        return None
    Ys, Zs, G = [], [], []
    for b in blocks:
        Z = np.column_stack([b["S"][:, 1:] if b["S"].shape[1] > 1
                             else np.zeros((b["S"].shape[0], 0)),
                             b["ylag"].reshape(-1, 1), b["xlag"]])
        Ys.append(b["dy"]); Zs.append(Z)
        G.append(np.array([b["unit"]] * b["dy"].size, dtype=object))
    Y = np.concatenate(Ys)
    Z = np.vstack(Zs)
    Gv = np.concatenate(G)
    Yd, Zd = Y.copy(), Z.copy()
    for u in np.unique(Gv):
        m = Gv == u
        Yd[m] -= Yd[m].mean()
        Zd[m] -= Zd[m].mean(axis=0)
    try:
        r = ols(Yd, Zd, has_const=False)
    except Exception:
        return None
    nu = len(np.unique(Gv))
    dfr = max(Y.size - Z.shape[1] - nu, 1)
    scale = r.df_resid / dfr
    cov = r.cov * scale
    ns = Z.shape[1] - 1 - k
    phi = float(r.beta[ns])
    beta = r.beta[ns + 1: ns + 1 + k]
    if abs(phi) < 1e-8:
        return None
    theta = -beta / phi
    # delta yontemi
    lr = []
    for j in range(k):
        gjac = np.zeros(Z.shape[1])
        gjac[ns] = beta[j] / phi ** 2
        gjac[ns + 1 + j] = -1.0 / phi
        v = float(gjac @ cov @ gjac)
        se = np.sqrt(max(v, 1e-300))
        t = theta[j] / se if se > 0 else np.nan
        lr.append({"name": pnl.indep[j], "B": f(theta[j]), "SE": f(se), "t": f(t),
                   "p": f(float(2 * stats.norm.sf(abs(t)))) if np.isfinite(t) else None,
                   "sig": star(float(2 * stats.norm.sf(abs(t))) if np.isfinite(t) else 1)})
    se_phi = float(np.sqrt(max(cov[ns, ns], 1e-300)))
    tphi = phi / se_phi if se_phi > 0 else np.nan
    return {"name": "Dinamik Sabit Etkiler (DFE) Tahmincisi",
            "N": nu, "p": p, "q": q, "longRun": lr,
            "ecm": {"name": "Hata Düzeltme Katsayısı (φ)", "B": f(phi),
                    "SE": f(se_phi), "t": f(tphi),
                    "p": f(float(2 * stats.norm.sf(abs(tphi)))) if np.isfinite(tphi) else None}}


def pooled_mean_group(pnl: Panel, p=1, q=1, start=None):
    """PMG (Pesaran, Shin & Smith, 1999) - yogunlastirilmis olabilirlik ile."""
    blocks = _ecm_blocks(pnl, p, q)
    k = len(pnl.indep)
    if len(blocks) < 3:
        return None

    def nll(theta):
        tot = 0.0
        for b in blocks:
            z = b["ylag"] - b["xlag"] @ theta
            Z = np.column_stack([b["S"], z.reshape(-1, 1)])
            try:
                bb = np.linalg.lstsq(Z, b["dy"], rcond=None)[0]
            except np.linalg.LinAlgError:
                return 1e12
            e = b["dy"] - Z @ bb
            n = e.size
            s2 = float(e @ e) / n
            if s2 <= 0 or not np.isfinite(s2):
                return 1e12
            tot += n * np.log(s2)
        return 0.5 * tot

    if start is None:
        mg = mean_group(pnl, p, q)
        start = np.asarray(mg["theta"], float) if mg else np.zeros(k)
    start = np.nan_to_num(np.asarray(start, float), nan=0.0,
                          posinf=0.0, neginf=0.0)
    res = optimize.minimize(nll, start, method="Nelder-Mead",
                            options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-10})
    if not np.all(np.isfinite(res.x)):
        return None
    theta = res.x

    # sayisal Hessian -> standart hatalar
    h = np.maximum(1e-4 * np.abs(theta), 1e-5)
    H = np.zeros((k, k))
    f0 = nll(theta)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = h[i]
            ej = np.zeros(k); ej[j] = h[j]
            fpp = nll(theta + ei + ej)
            fpm = nll(theta + ei - ej)
            fmp = nll(theta - ei + ej)
            fmm = nll(theta - ei - ej)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * h[i] * h[j])
    try:
        V = np.linalg.pinv(H)
    except np.linalg.LinAlgError:
        V = np.eye(k) * np.nan
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    t = np.divide(theta, se, out=np.full(k, np.nan), where=se > 0)
    pv = 2 * stats.norm.sf(np.abs(t))

    phis, sr = [], []
    for b in blocks:
        z = b["ylag"] - b["xlag"] @ theta
        Z = np.column_stack([b["S"], z.reshape(-1, 1)])
        bb = np.linalg.lstsq(Z, b["dy"], rcond=None)[0]
        phis.append(float(bb[-1]))
    ph = np.asarray(phis)
    pbar = float(ph.mean())
    pse = float(ph.std(ddof=1) / np.sqrt(ph.size)) if ph.size > 1 else np.nan
    pt = pbar / pse if pse and pse > 0 else np.nan
    return {
        "name": "Havuzlanmış Ortalama Grup (Pooled Mean Group, PMG) Tahmincisi",
        "N": len(blocks), "p": p, "q": q, "converged": bool(res.success),
        "logLik": f(-float(f0)),
        "longRun": [{"name": v, "B": f(theta[j]), "SE": f(se[j]), "t": f(t[j]),
                     "p": f(pv[j]), "sig": star(pv[j])} for j, v in enumerate(pnl.indep)],
        "ecm": {"name": "Hata Düzeltme Katsayısı (φ)", "B": f(pbar), "SE": f(pse),
                "t": f(pt),
                "p": f(float(2 * stats.norm.sf(abs(pt)))) if np.isfinite(pt) else None},
        "theta": theta.tolist(), "thetaVar": np.diag(V).tolist(),
    }


def hausman_mg_pmg(mg, pmg, indep):
    if not mg or not pmg:
        return None
    d = np.asarray(mg["theta"], float) - np.asarray(pmg["theta"], float)
    D = np.diag(np.asarray(mg["thetaVar"], float) -
                np.asarray(pmg["thetaVar"], float))
    w, V = np.linalg.eigh((D + D.T) / 2.0)
    tol = max(abs(w).max(), 1e-12) * 1e-8
    pos = w > tol
    if not pos.any():
        return None
    Di = (V[:, pos] / w[pos]) @ V[:, pos].T
    stat = float(d @ Di @ d)
    dfree = int(pos.sum())
    p = float(stats.chi2.sf(stat, dfree))
    return {"name": "Hausman Testi (MG vs. PMG)",
            "h0": "Uzun dönem katsayıları birimler arasında homojendir "
                  "(PMG etkindir)",
            "stat": f(stat), "df": dfree, "p": f(p),
            "decision": "MG (heterojen)" if p < 0.05 else "PMG (homojen uzun dönem)"}


# ==========================================================================

def run_all(pnl: Panel, gmm=True, ardl=True, ylag=1, max_inst=3,
            twostep=True, time_dummies=False, p=1, q=1, collapse=True):
    out = {}
    if gmm:
        try:
            out["gmm"] = arellano_bond(pnl, ylag, max_inst, twostep,
                                       time_dummies, collapse)
        except Exception as exc:
            out["gmm"] = None
            out["gmmError"] = str(exc)
    if ardl:
        mg = pmg = dfe = None
        try:
            mg = mean_group(pnl, p, q)
        except Exception as exc:
            out["mgError"] = str(exc)
        try:
            pmg = pooled_mean_group(pnl, p, q,
                                    start=mg["theta"] if mg else None)
        except Exception as exc:
            out["pmgError"] = str(exc)
        try:
            dfe = dynamic_fe(pnl, p, q)
        except Exception as exc:
            out["dfeError"] = str(exc)
        out["ardl"] = {"mg": mg, "pmg": pmg, "dfe": dfe,
                       "hausman": hausman_mg_pmg(mg, pmg, pnl.indep)}
    return out
