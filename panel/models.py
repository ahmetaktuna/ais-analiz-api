"""Temel panel modelleri: Havuzlanmis EKK, Sabit Etkiler, Rastgele Etkiler."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .prepare import Panel
from .utils import f, star

warnings.filterwarnings("ignore")

COV_LABEL = {
    "unadjusted": "Klasik (homoskedastik)",
    "robust": "White heteroskedasite-dirençli (HC1)",
    "clustered": "Birim bazında kümelenmiş (cluster-robust)",
    "kernel": "Driscoll-Kraay (yatay kesit bağımlılığına dirençli)",
}


def _mi(pnl: Panel):
    """linearmodels icin MultiIndex'li DataFrame."""
    d = pnl.df.copy()
    # linearmodels zaman indeksinin sayisal ya da tarih olmasini ister
    ts = sorted(pd.unique(d[pnl.time_var]))
    tmap = {t: i for i, t in enumerate(ts)}
    d["__t"] = d[pnl.time_var].map(tmap).astype("int64")
    d = d.set_index([pnl.id_var, "__t"])
    return d


def _coef_table(names, b, se, dfr, sd_x=None, sd_y=None, dist="t"):
    rows = []
    for j, nm in enumerate(names):
        bj, sj = float(b[j]), float(se[j])
        t = bj / sj if sj > 0 else np.nan
        if dist == "t":
            p = 2 * stats.t.sf(abs(t), max(dfr, 1)) if np.isfinite(t) else np.nan
            crit = stats.t.ppf(0.975, max(dfr, 1))
        else:
            p = 2 * stats.norm.sf(abs(t)) if np.isfinite(t) else np.nan
            crit = 1.959963985
        beta_std = None
        if sd_x is not None and sd_y and nm in sd_x and sd_y > 0:
            beta_std = f(bj * sd_x[nm] / sd_y)
        rows.append({
            "name": nm, "B": f(bj), "SE": f(sj), "t": f(t), "p": f(p),
            "beta": beta_std, "ciLow": f(bj - crit * sj), "ciHigh": f(bj + crit * sj),
            "sig": star(p),
        })
    return rows


def _sds(pnl: Panel):
    sd_x = {v: float(pnl.df[v].std(ddof=1)) for v in pnl.indep}
    sd_y = float(pnl.df[pnl.dep].std(ddof=1))
    return sd_x, sd_y


# --------------------------------------------------------------------------

def fit_all(pnl: Panel, effects="entity", cov_type="clustered"):
    """
    Havuzlanmis EKK, Sabit Etkiler ve Rastgele Etkiler modellerini tahmin eder,
    model secim testlerini (F, LM, Hausman) uygular.
    """
    from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects

    d = _mi(pnl)
    y = d[pnl.dep]
    X = d[pnl.indep].copy()
    X.insert(0, "Sabit", 1.0)
    names = ["Sabit"] + pnl.indep
    sd_x, sd_y = _sds(pnl)

    ent = effects in ("entity", "twoway")
    tim = effects in ("time", "twoway")

    def _cov_kwargs(ct):
        if ct == "clustered":
            return {"cov_type": "clustered", "cluster_entity": True}
        if ct == "kernel":
            return {"cov_type": "kernel"}
        if ct == "robust":
            return {"cov_type": "robust"}
        return {"cov_type": "unadjusted"}

    out = {"effects": effects, "covType": cov_type,
           "covLabel": COV_LABEL.get(cov_type, cov_type)}

    # ---- Havuzlanmis EKK -------------------------------------------------
    pooled_mod = PooledOLS(y, X)
    pooled = pooled_mod.fit(**_cov_kwargs(cov_type))
    out["pooled"] = {
        "label": "Havuzlanmış EKK (Pooled OLS)",
        "coeffs": _coef_table(names, pooled.params.values, pooled.std_errors.values,
                              pooled.df_resid, sd_x, sd_y),
        "r2": f(pooled.rsquared), "r2Adj": f(1 - (1 - pooled.rsquared) *
                                             (pooled.nobs - 1) / max(pooled.df_resid, 1)),
        "F": f(pooled.f_statistic.stat), "pF": f(pooled.f_statistic.pval),
        "dfModel": int(pooled.df_model - 1), "dfResid": int(pooled.df_resid),
        "nobs": int(pooled.nobs),
        "loglik": f(getattr(pooled, "loglik", np.nan)),
        "rmse": f(np.sqrt(float((pooled.resids.values ** 2).mean()))),
    }

    # ---- Sabit Etkiler ---------------------------------------------------
    fe_mod = PanelOLS(y, X, entity_effects=ent, time_effects=tim,
                      drop_absorbed=True)
    fe = fe_mod.fit(**_cov_kwargs(cov_type))
    fe_names = list(fe.params.index)
    out["fe"] = {
        "label": ("Sabit Etkiler (Fixed Effects)" if effects == "entity" else
                  "İki Yönlü Sabit Etkiler" if effects == "twoway" else
                  "Zaman Sabit Etkileri"),
        "coeffs": _coef_table(fe_names, fe.params.values, fe.std_errors.values,
                              fe.df_resid, sd_x, sd_y),
        "r2": f(fe.rsquared), "r2Within": f(fe.rsquared_within),
        "r2Between": f(fe.rsquared_between), "r2Overall": f(fe.rsquared_overall),
        "F": f(fe.f_statistic.stat), "pF": f(fe.f_statistic.pval),
        "dfModel": int(fe.df_model), "dfResid": int(fe.df_resid),
        "nobs": int(fe.nobs),
        "rmse": f(np.sqrt(float((fe.resids.values ** 2).mean()))),
        "sigmaU": f(float(np.std(fe.estimated_effects.values.ravel(), ddof=1))),
        "sigmaE": f(float(np.std(fe.resids.values.ravel(), ddof=1))),
    }
    su = out["fe"]["sigmaU"] or 0.0
    se_ = out["fe"]["sigmaE"] or 0.0
    out["fe"]["rho"] = f(su ** 2 / (su ** 2 + se_ ** 2)) if (su + se_) > 0 else None

    # F testi: sabit etkiler anlamli mi? (Havuzlanmis vs Sabit Etkiler)
    try:
        fp = fe_mod.fit(cov_type="unadjusted").f_pooled
        out["fTest"] = {
            "name": "F Testi (Havuzlanmış EKK vs. Sabit Etkiler)",
            "h0": "Birim etkileri sıfırdır (havuzlanmış model uygundur)",
            "stat": f(fp.stat), "df1": int(fp.df), "df2": int(fp.df_denom),
            "p": f(fp.pval),
            "decision": "Sabit Etkiler" if fp.pval < 0.05 else "Havuzlanmış EKK",
        }
    except Exception:
        out["fTest"] = None

    # ---- Rastgele Etkiler ------------------------------------------------
    re = None
    try:
        re_mod = RandomEffects(y, X)
        re = re_mod.fit(**_cov_kwargs("clustered" if cov_type == "kernel" else cov_type))
        vd = re.variance_decomposition
        out["re"] = {
            "label": "Rastgele Etkiler (Random Effects, GLS)",
            "coeffs": _coef_table(names, re.params.values, re.std_errors.values,
                                  re.df_resid, sd_x, sd_y),
            "r2": f(re.rsquared), "r2Within": f(re.rsquared_within),
            "r2Between": f(re.rsquared_between), "r2Overall": f(re.rsquared_overall),
            "F": f(re.f_statistic.stat), "pF": f(re.f_statistic.pval),
            "dfModel": int(re.df_model - 1), "dfResid": int(re.df_resid),
            "nobs": int(re.nobs),
            "theta": f(float(np.mean(np.asarray(re.theta).ravel()))),
            "varEffects": f(float(vd.get("Effects", np.nan))),
            "varResidual": f(float(vd.get("Residual", np.nan))),
            "rho": f(float(vd.get("Percent due to Effects", np.nan))),
        }
    except Exception as exc:  # pragma: no cover
        out["re"] = None
        out["reError"] = str(exc)

    # ---- Breusch-Pagan LM (Havuzlanmis vs Rastgele Etkiler) --------------
    out["lmTest"] = breusch_pagan_lm(pnl, pooled.resids.values.ravel())

    # ---- Hausman ---------------------------------------------------------
    out["hausman"] = None
    if re is not None:
        out["hausman"] = hausman(fe, re, pnl.indep)

    # ---- Model secimi ----------------------------------------------------
    out["selection"] = _select_model(out)

    # nihai model
    key = out["selection"]["preferred"]
    out["final"] = dict(out[key]) if out.get(key) else dict(out["fe"])
    out["final"]["key"] = key
    return out, {"pooled": pooled, "fe": fe, "re": re}


def _psig(v, alpha=0.05):
    """p-degeri anlamli mi? (p = 0.0 degerinin falsy olmasina karsi guvenli)"""
    try:
        return v is not None and float(v) < alpha
    except (TypeError, ValueError):
        return False


def _select_model(res):
    ftest, lm, hs = res.get("fTest"), res.get("lmTest"), res.get("hausman")
    f_sig = _psig(ftest.get("p")) if ftest else False
    lm_sig = _psig(lm.get("p")) if lm else False
    steps = []
    if ftest:
        steps.append(
            "F testi: birim etkileri "
            + ("anlamlı → Havuzlanmış EKK reddedilir."
               if f_sig else "anlamsız → Havuzlanmış EKK yeterlidir."))
    if lm:
        steps.append(
            "Breusch-Pagan LM testi: birim etkilerinin varyansı "
            + ("anlamlı → Rastgele Etkiler, Havuzlanmış EKK'ya tercih edilir."
               if lm_sig else "anlamsız → Havuzlanmış EKK yeterlidir."))

    if not f_sig and not lm_sig:
        pref, why = "pooled", ("Hem F hem de Breusch-Pagan LM testi anlamsız "
                               "olduğundan Havuzlanmış EKK modeli uygundur.")
    else:
        if hs and hs.get("p") is not None:
            if _psig(hs["p"]):
                pref = "fe"
                why = ("Hausman testi anlamlı bulunduğundan birim etkiler ile "
                       "açıklayıcı değişkenler arasında korelasyon olduğu kabul "
                       "edilmiş ve tutarlı tahminci olan Sabit Etkiler modeli "
                       "tercih edilmiştir.")
            else:
                pref = "re"
                why = ("Hausman testi anlamsız bulunduğundan hem tutarlı hem de "
                       "etkin olan Rastgele Etkiler tahmincisi tercih edilmiştir.")
            steps.append(why)
        else:
            pref, why = "fe", "Hausman testi hesaplanamadığından Sabit Etkiler modeli esas alınmıştır."
    label = {"pooled": "Havuzlanmış EKK", "fe": "Sabit Etkiler",
             "re": "Rastgele Etkiler"}[pref]
    return {"preferred": pref, "preferredLabel": label, "reason": why, "steps": steps}


# --------------------------------------------------------------------------

def breusch_pagan_lm(pnl: Panel, resid):
    """
    Breusch & Pagan (1980) Lagrange Carpani testi; dengesiz panel icin
    Baltagi & Li (1990) genellemesi. H0: Var(u_i) = 0 (birim etkisi yok).
    """
    df = pnl.df.copy()
    df["__e"] = np.asarray(resid, float)
    g = df.groupby(pnl.id_var, observed=True)["__e"]
    sums = g.sum().to_numpy(float)
    Ti = g.size().to_numpy(float)
    e2 = float((df["__e"].to_numpy(float) ** 2).sum())
    if e2 <= 0:
        return None
    n = float(Ti.sum())
    denom = float((Ti * (Ti - 1.0)).sum())
    if denom <= 0:
        return None
    A = float((sums ** 2).sum()) / e2
    lm = (n ** 2 / (2.0 * denom)) * (A - 1.0) ** 2
    p = float(stats.chi2.sf(lm, 1))
    honda = np.sqrt(n ** 2 / (2.0 * denom)) * (A - 1.0)
    return {
        "name": "Breusch-Pagan LM Testi (Havuzlanmış EKK vs. Rastgele Etkiler)",
        "h0": "Birim etkilerinin varyansı sıfırdır (Var(uᵢ) = 0)",
        "stat": f(lm), "df": 1, "p": f(p),
        "honda": f(honda), "hondaP": f(float(stats.norm.sf(honda))),
        "decision": "Rastgele Etkiler" if p < 0.05 else "Havuzlanmış EKK",
    }


def hausman(fe_res, re_res, indep):
    """Hausman (1978) spesifikasyon testi (yalnizca ortak egim katsayilari)."""
    common = [v for v in indep if v in fe_res.params.index and v in re_res.params.index]
    if not common:
        return None
    b = fe_res.params[common].to_numpy(float)
    B = re_res.params[common].to_numpy(float)
    Vb = fe_res.cov.loc[common, common].to_numpy(float)
    VB = re_res.cov.loc[common, common].to_numpy(float)
    d = b - B
    D = Vb - VB
    # PSD olmayan farklar icin genellestirilmis ters
    try:
        w, V = np.linalg.eigh((D + D.T) / 2.0)
    except np.linalg.LinAlgError:
        return None
    tol = max(abs(w).max(), 1e-12) * 1e-8
    pos = w > tol
    if not pos.any():
        return None
    Dinv = (V[:, pos] / w[pos]) @ V[:, pos].T
    stat = float(d @ Dinv @ d)
    dfree = int(pos.sum())
    p = float(stats.chi2.sf(stat, dfree))
    return {
        "name": "Hausman Testi (Sabit Etkiler vs. Rastgele Etkiler)",
        "h0": "Birim etkileri ile açıklayıcı değişkenler ilişkisizdir "
              "(Rastgele Etkiler tutarlıdır)",
        "stat": f(stat), "df": dfree, "p": f(p),
        "decision": "Sabit Etkiler" if p < 0.05 else "Rastgele Etkiler",
        "psdWarning": bool(dfree < len(common)),
        "detail": [{"variable": v, "bFE": f(b[i]), "bRE": f(B[i]),
                    "diff": f(d[i])} for i, v in enumerate(common)],
    }


def robust_variants(pnl: Panel, effects="entity"):
    """Nihai model icin farkli standart hata secenekleri (karsilastirma tablosu)."""
    from linearmodels.panel import PanelOLS
    d = _mi(pnl)
    y = d[pnl.dep]
    X = d[pnl.indep].copy()
    X.insert(0, "Sabit", 1.0)
    ent = effects in ("entity", "twoway")
    tim = effects in ("time", "twoway")
    mod = PanelOLS(y, X, entity_effects=ent, time_effects=tim, drop_absorbed=True)
    out = []
    specs = [("unadjusted", {"cov_type": "unadjusted"}),
             ("robust", {"cov_type": "robust"}),
             ("clustered", {"cov_type": "clustered", "cluster_entity": True}),
             ("kernel", {"cov_type": "kernel"})]
    for key, kw in specs:
        try:
            r = mod.fit(**kw)
            out.append({
                "key": key, "label": COV_LABEL[key],
                "coeffs": _coef_table(list(r.params.index), r.params.values,
                                      r.std_errors.values, r.df_resid),
            })
        except Exception:
            continue
    return out
