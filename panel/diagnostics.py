"""Panel tanilayici testler: degisen varyans, otokorelasyon, yatay kesit bagimliligi."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .prepare import Panel
from .utils import add_const, f, ols


# --------------------------------------------------------------------------
# Cok dogrusal baglanti
# --------------------------------------------------------------------------

def vif(pnl: Panel):
    rows = []
    k = len(pnl.indep)
    if k == 1:
        return [{"variable": pnl.indep[0], "vif": 1.0, "tolerance": 1.0}]
    Xall = pnl.df[pnl.indep].to_numpy(float)
    for j, v in enumerate(pnl.indep):
        yj = Xall[:, j]
        Xj = np.delete(Xall, j, axis=1)
        try:
            r = ols(yj, add_const(Xj))
            r2 = r.r2
        except Exception:
            r2 = np.nan
        if not np.isfinite(r2) or r2 >= 0.999999:
            rows.append({"variable": v, "vif": None, "tolerance": None})
        else:
            rows.append({"variable": v, "vif": f(1.0 / (1.0 - r2)),
                         "tolerance": f(1.0 - r2)})
    return rows


# --------------------------------------------------------------------------
# Degisen varyans
# --------------------------------------------------------------------------

def modified_wald(pnl: Panel, resid):
    """
    Greene (2000) Modified Wald testi - Sabit Etkiler modelinde gruplar arasi
    degisen varyans. H0: sigma_i^2 = sigma^2 (tum i icin).
    """
    df = pnl.df.copy()
    df["__e"] = np.asarray(resid, float)
    e_all = df["__e"].to_numpy(float)
    s2 = float((e_all ** 2).sum()) / e_all.size
    stat, used = 0.0, 0
    for _, g in df.groupby(pnl.id_var, observed=True):
        e = g["__e"].to_numpy(float)
        Ti = e.size
        if Ti < 3:
            continue
        s2i = float((e ** 2).mean())
        Vi = float(((e ** 2 - s2i) ** 2).sum()) / (Ti * (Ti - 1.0))
        if Vi <= 0:
            continue
        stat += (s2i - s2) ** 2 / Vi
        used += 1
    if used < 2:
        return None
    p = float(stats.chi2.sf(stat, used))
    return {"name": "Modified Wald Testi (gruplar arası değişen varyans)",
            "h0": "Birimler arası hata varyansları eşittir (homoskedastisite)",
            "stat": f(stat), "df": used, "p": f(p),
            "problem": bool(p < 0.05)}


def breusch_pagan_het(pnl: Panel, resid):
    """Breusch-Pagan / Cook-Weisberg degisen varyans testi."""
    e = np.asarray(resid, float)
    X = add_const(pnl.df[pnl.indep].to_numpy(float))
    g = e ** 2 / float((e ** 2).mean())
    try:
        r = ols(g, X)
    except Exception:
        return None
    lm = 0.5 * float(((r.fitted - g.mean()) ** 2).sum())
    dfree = X.shape[1] - 1
    p = float(stats.chi2.sf(lm, dfree))
    return {"name": "Breusch-Pagan / Cook-Weisberg Testi",
            "h0": "Hata varyansı sabittir (homoskedastisite)",
            "stat": f(lm), "df": dfree, "p": f(p), "problem": bool(p < 0.05)}


# --------------------------------------------------------------------------
# Otokorelasyon
# --------------------------------------------------------------------------

def wooldridge_ar1(pnl: Panel):
    """
    Wooldridge (2002) / Drukker (2003) panel otokorelasyon testi.
    Birinci fark artiklarini kendi gecikmesine regresyon edip
    H0: rho = -0.5 sinanir (kume-dirençli varyans ile).
    """
    d = pnl.df
    parts_dy, parts_dx, parts_id = [], [], []
    for u, g in d.groupby(pnl.id_var, sort=False, observed=True):
        if len(g) < 3:
            continue
        dy = np.diff(g[pnl.dep].to_numpy(float))
        dx = np.diff(g[pnl.indep].to_numpy(float), axis=0)
        parts_dy.append(dy)
        parts_dx.append(dx)
        parts_id.append(np.array([u] * dy.size, dtype=object))
    if not parts_dy:
        return None
    dy = np.concatenate(parts_dy)
    dx = np.vstack(parts_dx)
    ids = np.concatenate(parts_id)
    if dy.size <= dx.shape[1] + 2:
        return None
    try:
        r = ols(dy, dx, has_const=False)
    except Exception:
        return None
    e = r.resid

    # e_it ~ e_{i,t-1}
    ys, xs, cl = [], [], []
    pos = 0
    for arr in parts_dy:
        n = arr.size
        ei = e[pos:pos + n]
        uid = ids[pos]
        pos += n
        if n < 2:
            continue
        ys.append(ei[1:])
        xs.append(ei[:-1])
        cl.append(np.array([uid] * (n - 1), dtype=object))
    if not ys:
        return None
    Y = np.concatenate(ys)
    Xv = np.concatenate(xs).reshape(-1, 1)
    C = np.concatenate(cl)
    if Y.size < 5:
        return None
    r2_ = ols(Y, Xv, has_const=False)
    rho = float(r2_.beta[0])

    # kume-direncli varyans
    XtXi = r2_.XtXi
    meat = np.zeros((1, 1))
    for u in pd.unique(C):
        m = C == u
        xu = Xv[m]
        eu = r2_.resid[m]
        s = xu.T @ eu
        meat += np.outer(s, s)
    G = len(pd.unique(C))
    nG = Y.size
    scale = (G / max(G - 1.0, 1.0)) * ((nG - 1.0) / max(nG - 1.0, 1.0))
    V = scale * XtXi @ meat @ XtXi
    se = float(np.sqrt(max(V[0, 0], 1e-300)))
    Fstat = ((rho + 0.5) / se) ** 2 if se > 0 else np.nan
    p = float(stats.f.sf(Fstat, 1, max(G - 1, 1))) if np.isfinite(Fstat) else np.nan
    return {"name": "Wooldridge Testi (birinci derece otokorelasyon)",
            "h0": "Birinci derece otokorelasyon yoktur (ρ = −0,5)",
            "rho": f(rho), "stat": f(Fstat), "df1": 1, "df2": int(max(G - 1, 1)),
            "p": f(p), "problem": bool(np.isfinite(p) and p < 0.05)}


def panel_dw(pnl: Panel, resid):
    """Bhargava, Franzini & Narendranathan (1982) panel Durbin-Watson."""
    df = pnl.df.copy()
    df["__e"] = np.asarray(resid, float)
    num = den = 0.0
    for _, g in df.groupby(pnl.id_var, observed=True):
        e = g["__e"].to_numpy(float)
        if e.size >= 2:
            num += float(((e[1:] - e[:-1]) ** 2).sum())
        den += float((e ** 2).sum())
    if den <= 0:
        return None
    dw = num / den
    return {"name": "Panel Durbin-Watson (Bhargava vd., 1982)",
            "stat": f(dw),
            "note": "2'ye yakın değerler otokorelasyon olmadığına işaret eder.",
            "problem": bool(dw < 1.5 or dw > 2.5)}


# --------------------------------------------------------------------------
# Yatay kesit bagimliligi
# --------------------------------------------------------------------------

def _resid_matrix(pnl: Panel, resid):
    d = pnl.df.copy()
    d["__e"] = np.asarray(resid, float)
    return d.pivot_table(index=pnl.time_var, columns=pnl.id_var,
                         values="__e", aggfunc="first").sort_index()


def cross_section_dependence(pnl: Panel, resid, k=1):
    """Pesaran CD, Breusch-Pagan LM ve olcekli LM testleri."""
    M = _resid_matrix(pnl, resid)
    cols = list(M.columns)
    N = len(cols)
    if N < 2:
        return None
    A = M.to_numpy(float)
    pairs, cd_sum, lm_sum, slm_sum, npairs = [], 0.0, 0.0, 0.0, 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            xi, xj = A[:, i], A[:, j]
            m = np.isfinite(xi) & np.isfinite(xj)
            Tij = int(m.sum())
            if Tij < 4:
                continue
            a, b = xi[m], xj[m]
            a = a - a.mean()
            b = b - b.mean()
            den = np.sqrt(float(a @ a) * float(b @ b))
            if den <= 0:
                continue
            rho = float(a @ b) / den
            cd_sum += np.sqrt(Tij) * rho
            lm_sum += Tij * rho ** 2
            slm_sum += (Tij - k) * rho ** 2 - 1.0
            npairs += 1
    if npairs < 1:
        return None
    cd = np.sqrt(2.0 / (N * (N - 1.0))) * cd_sum
    cd_p = float(2 * stats.norm.sf(abs(cd)))
    lm = lm_sum
    lm_df = npairs
    lm_p = float(stats.chi2.sf(lm, lm_df))
    slm = np.sqrt(1.0 / (N * (N - 1.0))) * slm_sum
    slm_p = float(2 * stats.norm.sf(abs(slm)))
    return {
        "name": "Yatay Kesit Bağımlılığı Testleri",
        "h0": "Birimler arasında yatay kesit bağımlılığı yoktur",
        "tests": [
            {"test": "Pesaran CD", "stat": f(cd), "p": f(cd_p),
             "dist": "N(0,1)", "problem": bool(cd_p < 0.05)},
            {"test": "Breusch-Pagan LM", "stat": f(lm), "df": lm_df, "p": f(lm_p),
             "dist": f"χ²({lm_df})", "problem": bool(lm_p < 0.05),
             "note": "T > N olduğunda güvenilirdir."},
            {"test": "Pesaran ölçekli LM", "stat": f(slm), "p": f(slm_p),
             "dist": "N(0,1)", "problem": bool(slm_p < 0.05)},
        ],
        "problem": bool(cd_p < 0.05),
        "npairs": npairs, "N": N,
    }


# --------------------------------------------------------------------------
# Normallik + toplu calistirici
# --------------------------------------------------------------------------

def normality(resid):
    e = np.asarray(resid, float)
    e = e[np.isfinite(e)]
    if e.size < 8:
        return None
    jb, p = stats.jarque_bera(e)[:2]
    return {"name": "Jarque-Bera Normallik Testi",
            "h0": "Hatalar normal dağılmaktadır",
            "stat": f(jb), "df": 2, "p": f(p),
            "skewness": f(stats.skew(e)), "kurtosis": f(stats.kurtosis(e, fisher=False)),
            "problem": bool(p < 0.05)}


def run_all(pnl: Panel, fe_resid, pooled_resid):
    k = len(pnl.indep) + 1
    het_w = modified_wald(pnl, fe_resid)
    het_bp = breusch_pagan_het(pnl, pooled_resid)
    auto = wooldridge_ar1(pnl)
    dw = panel_dw(pnl, fe_resid)
    csd = cross_section_dependence(pnl, fe_resid, k=k)
    norm = normality(fe_resid)
    v = vif(pnl)

    het_prob = bool((het_w and het_w["problem"]) or (het_bp and het_bp["problem"]))
    auto_prob = bool(auto and auto["problem"])
    csd_prob = bool(csd and csd["problem"])

    if csd_prob:
        rec, rec_key = ("Driscoll-Kraay dirençli standart hatalar", "kernel")
    elif het_prob or auto_prob:
        rec, rec_key = ("Birim bazında kümelenmiş (cluster-robust) standart hatalar",
                        "clustered")
    else:
        rec, rec_key = ("Klasik standart hatalar", "unadjusted")

    issues = []
    if het_prob:
        issues.append("değişen varyans (heteroskedasite)")
    if auto_prob:
        issues.append("otokorelasyon")
    if csd_prob:
        issues.append("yatay kesit bağımlılığı")

    return {
        "heteroskedasticity": {"modifiedWald": het_w, "breuschPagan": het_bp,
                               "problem": het_prob},
        "autocorrelation": {"wooldridge": auto, "panelDW": dw, "problem": auto_prob},
        "crossSectionDependence": csd,
        "normality": norm,
        "multicollinearity": {
            "vif": v,
            "problem": bool(any((r.get("vif") or 0) > 10 for r in v)),
            "maxVif": f(max([r.get("vif") or 0 for r in v]) if v else 0),
        },
        "issues": issues,
        "recommendedSE": rec,
        "recommendedSEKey": rec_key,
    }
