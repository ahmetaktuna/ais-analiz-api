"""
AIS Akademi - Istatistiksel Analiz API
======================================
Uc noktalar:
    GET  /ping
    POST /analyze          -> coklu dogrusal regresyon
    POST /ttest-multiple   -> bagimsiz orneklem t-testi (coklu degisken)
    POST /anova-multiple   -> tek yonlu ANOVA + post-hoc (coklu degisken)
    POST /normality        -> normallik sinamasi ve betimsel istatistikler
    POST /mannwhitney-multiple  -> Mann-Whitney U (t-testinin non-parametrik karsiligi)
    POST /kruskal-multiple      -> Kruskal-Wallis H + Dunn (ANOVA'nin karsiligi)
    POST /pairedttest-multiple  -> bagimli (eslestirilmis) orneklem t-testi
    POST /wilcoxon-multiple     -> Wilcoxon isaretli siralar (bagimli t'nin karsiligi)
    POST /correlation-multiple  -> Pearson / Spearman korelasyon matrisi
    POST /roc-multiple          -> ROC egrisi analizi (AUC, DeLong, kesme noktasi)
    POST /logistic              -> lojistik regresyon (ikili / sirali / cok kategorili)

Panel veri ve zaman serisi modulleri ayri router'lar olarak eklenir.

YANIT SOZLESMESI: Mevcut sayfalarin okudugu tum alan adlari korunmustur.
Yeni alanlar yalnizca eklenmistir; hicbir alan kaldirilmamis veya yeniden
adlandirilmamistir.
"""
from __future__ import annotations

import gc
import itertools
import logging
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

log = logging.getLogger("ais-api")

# --------------------------------------------------------------------------
# Kaynak sinirlari
# --------------------------------------------------------------------------
# Render Free ornegi 512 MB bellekle calisir ve bu surec panel/zaman serisi
# modulleriyle AYNI surectir. Asiri buyuk bir yukleme surecin oldurulmesine,
# dolayisiyla TUM analiz sayfalarinin birden kesilmesine yol acar. Bu yuzden
# istekler daha ayristirilmadan once boyut acisindan reddedilir.
MAX_ROWS = int(os.getenv("AIS_MAX_ROWS", "200000"))
MAX_CELLS = int(os.getenv("AIS_MAX_CELLS", "4000000"))
MAX_VARS = int(os.getenv("AIS_MAX_VARS", "200"))

app = FastAPI(title="AIS Akademi Analiz API", version="2.0.0")

# allow_credentials=False: sayfalar cerez/kimlik gondermiyor. Joker kokenle
# birlikte kimlik bilgisi izni CORS sartnamesine aykiridir; kapatmak mevcut
# istekleri etkilemez, tarayici tarafindaki belirsizligi ortadan kaldirir.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,
)

# === PANEL VERI ANALIZI MODULU ===
# Ekledigi uc noktalar: /panel-health, /panel-detect, /panel-analyze
from panel_api import router as panel_router  # noqa: E402
app.include_router(panel_router)

# === ZAMAN SERISI ANALIZI MODULU ===
# Ekledigi uc noktalar: /ts-health, /ts-detect, /ts-analyze
from ts_api import router as ts_router  # noqa: E402
app.include_router(ts_router)
# ===================================


# --------------------------------------------------------------------------
# Ortak yardimcilar
# --------------------------------------------------------------------------

def _f(x, default=0.0) -> float:
    """NaN/Inf icermeyen bir float dondurur (JSON'da NaN gecersizdir)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _fo(x) -> Optional[float]:
    """Sonlu degilse None dondurur."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _guard(data, variables=()) -> Optional[dict]:
    """Istek boyutunu dogrular; sorun varsa hata sozlugu dondurur."""
    if not isinstance(data, list) or not data:
        return {"error": "Veri gönderilmedi."}
    n = len(data)
    if n > MAX_ROWS:
        return {"error": f"Veri seti çok büyük ({n:,} satır). "
                         f"En fazla {MAX_ROWS:,} satır işlenebilir."}
    ncol = len(data[0]) if isinstance(data[0], dict) else 0
    if n * max(ncol, 1) > MAX_CELLS:
        return {"error": f"Veri seti çok büyük ({n:,} satır × {ncol} sütun). "
                         f"Lütfen yalnızca analize girecek sütunları yükleyiniz."}
    if len(variables) > MAX_VARS:
        return {"error": f"Aynı anda en fazla {MAX_VARS} değişken analiz edilebilir."}
    return None


def _frame(data, cols) -> pd.DataFrame:
    """Yalnizca gerekli sutunlari alir; tum cerceveyi kopyalamaz."""
    df = pd.DataFrame(data)
    keep, seen = [], set()
    for c in cols:                      # sirayi koru, yinelenenleri at
        if c in df.columns and c not in seen:
            keep.append(c)
            seen.add(c)
    return df[keep]


def _numeric(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _missing(df: pd.DataFrame, cols) -> Optional[dict]:
    absent = [c for c in cols if c not in df.columns]
    if absent:
        return {"error": "Şu sütunlar veri setinde bulunamadı: " + ", ".join(absent)}
    return None


# --------------------------------------------------------------------------
# Istek modelleri
# --------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    depVar: str
    indepVars: List[str]
    data: List[Dict[str, Any]]


class MultipleTTestRequest(BaseModel):
    depVars: List[str]
    groupVar: str
    data: List[Dict[str, Any]]


class MultipleANOVARequest(BaseModel):
    depVars: List[str]
    groupVar: str
    postHoc: str
    data: List[Dict[str, Any]]


class NormalityRequest(BaseModel):
    depVars: List[str]
    data: List[Dict[str, Any]]


@app.get("/ping")
def ping():
    return {"status": "Uyanığım ve analize hazırım!"}


# ==========================================================================
# REGRESYON
# ==========================================================================

@app.post("/analyze")
def analyze(req: AnalysisRequest):
    try:
        cols = [req.depVar] + list(req.indepVars)
        bad = _guard(req.data, cols)
        if bad:
            return bad
        if not req.indepVars:
            return {"error": "En az bir bağımsız değişken seçmelisiniz."}
        if req.depVar in req.indepVars:
            return {"error": "Bağımlı değişken aynı zamanda bağımsız değişken olamaz."}

        df = _frame(req.data, cols)
        bad = _missing(df, cols)
        if bad:
            return bad

        indep = [c for c in dict.fromkeys(req.indepVars)]   # yinelenenleri at
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        n_before = len(df)
        df = df.dropna(subset=cols)
        n = len(df)
        k = len(indep)

        if n < k + 2:
            return {"error": f"Geçerli veri sayısı yetersiz (n = {n}). "
                             f"{k} bağımsız değişken için en az {k + 2} satır gereklidir."}

        # sabit sutun kontrolu: OLS'i NaN'a dusuren en yaygin sebep
        const_cols = [c for c in indep if df[c].nunique(dropna=True) < 2]
        if const_cols:
            return {"error": "Şu değişken(ler) tek bir değerden oluşuyor ve "
                             "regresyona giremez: " + ", ".join(const_cols)}
        if df[req.depVar].nunique(dropna=True) < 2:
            return {"error": "Bağımlı değişken tek bir değerden oluşuyor."}

        Y = df[req.depVar]
        X = sm.add_constant(df[indep], has_constant="add")

        # Tam coklu dogrusal baglanti kontrolu. statsmodels bu durumda hata
        # vermez; sozde-ters ile KEYFI bir katsayi kumesi dondurur. Bu da
        # kullaniciya anlamli gorunen ama yorumlanamayan bir tablo uretir.
        Xm = X.to_numpy(dtype=float)
        if np.linalg.matrix_rank(Xm) < Xm.shape[1]:
            pair = ""
            if k > 1:
                cm = df[indep].corr().abs()
                arr = np.array(cm.to_numpy(dtype=float), copy=True)
                np.fill_diagonal(arr, 0.0)
                if np.isfinite(arr).any() and float(np.nanmax(arr)) > 0.999:
                    i, j = np.unravel_index(int(np.nanargmax(arr)), arr.shape)
                    pair = f" (özellikle {cm.index[i]} ve {cm.columns[j]})"
            return {"error": "Bağımsız değişkenleriniz arasında tam doğrusal ilişki "
                             f"bulunduğu için model tahmin edilemedi{pair}. "
                             "Bu değişkenlerden birini modelden çıkarınız."}

        model = sm.OLS(Y, X).fit()

        dropped = [v for v in indep if not np.isfinite(model.params.get(v, np.nan))]
        if dropped:
            return {"error": "Şu değişken(ler) için katsayı hesaplanamadı: "
                             + ", ".join(dropped)
                             + ". Bu değişkenleri modelden çıkarınız."}

        # --- VIF ve tolerans (sabit terim icin None) ---
        vifs: List[Optional[float]] = [None]
        tols: List[Optional[float]] = [None]
        if k == 1:
            vifs.append(1.0)
            tols.append(1.0)
        else:
            Xv = X.to_numpy(dtype=float)
            for i in range(1, Xv.shape[1]):
                try:
                    v = float(variance_inflation_factor(Xv, i))
                except Exception:
                    v = float("inf")
                if not math.isfinite(v) or v <= 0:
                    vifs.append(999.0)
                    tols.append(0.001)
                else:
                    vifs.append(v)
                    tols.append(1.0 / v)

        # --- standartlastirilmis katsayilar ---
        sd_y = float(Y.std(ddof=1))
        if not math.isfinite(sd_y) or sd_y == 0:
            sd_y = 1.0
        betas: List[Optional[float]] = [None]
        for v in indep:
            sd_x = float(df[v].std(ddof=1))
            if not math.isfinite(sd_x):
                sd_x = 0.0
            betas.append(_f(model.params[v] * (sd_x / sd_y)))

        ci = model.conf_int(alpha=0.05)
        coeffData = []
        for i, v in enumerate(["const"] + indep):
            coeffData.append({
                "name": "Sabit Terim" if v == "const" else v,
                "B": _f(model.params[v]),
                "SE": _f(model.bse[v]),
                "Beta": betas[i],
                "t": _f(model.tvalues[v]),
                "p": _f(model.pvalues[v], 1.0),
                "Tol": tols[i],
                "VIF": vifs[i],
                # --- yeni alanlar ---
                "ciLow": _f(ci.loc[v, 0]),
                "ciHigh": _f(ci.loc[v, 1]),
            })

        # --- yeni: model tanilari ---
        resid = np.asarray(model.resid, dtype=float)
        diagnostics: Dict[str, Any] = {}
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp = het_breuschpagan(resid, X.to_numpy(dtype=float))
            diagnostics["breuschPagan"] = {"stat": _f(bp[0]), "p": _f(bp[1], 1.0),
                                           "df": int(k)}
        except Exception:
            pass
        try:
            jb = stats.jarque_bera(resid)
            diagnostics["jarqueBera"] = {"stat": _f(jb.statistic),
                                         "p": _f(jb.pvalue, 1.0)}
        except Exception:
            pass
        if 3 <= n <= 5000:
            try:
                sw = stats.shapiro(resid)
                diagnostics["shapiro"] = {"W": _f(sw.statistic),
                                          "p": _f(sw.pvalue, 1.0)}
            except Exception:
                pass
        try:                                    # en buyuk kosul indeksi
            Xs = X.to_numpy(dtype=float)
            norms = np.sqrt((Xs ** 2).sum(axis=0))
            norms[norms == 0] = 1.0
            sv = np.linalg.svd(Xs / norms, compute_uv=False)
            if sv.min() > 0:
                diagnostics["maxConditionIndex"] = _f(sv.max() / sv.min())
        except Exception:
            pass

        return {
            # --- mevcut alanlar (degismedi) ---
            "n": n, "k": k,
            "R2": _f(model.rsquared),
            "adjR2": _f(model.rsquared_adj),
            "F": _f(model.fvalue),
            "df_model": _f(model.df_model),
            "df_error": _f(model.df_resid),
            "p_F": _f(model.f_pvalue, 1.0),
            "DW": _f(durbin_watson(resid)),
            "coeffData": coeffData,
            "depVar": req.depVar,
            "indepVars": indep,
            # --- yeni alanlar ---
            "nDropped": int(n_before - n),
            "seEstimate": _f(np.sqrt(model.mse_resid)),
            "R": _f(math.sqrt(max(model.rsquared, 0.0))),
            "aic": _f(model.aic),
            "bic": _f(model.bic),
            "diagnostics": diagnostics,
        }
    except Exception as e:
        log.exception("analyze failed")
        return {"error": f"Regresyon Hatası: {e}"}
    finally:
        gc.collect()


# ==========================================================================
# BAGIMSIZ ORNEKLEM t-TESTI
# ==========================================================================

def _sorted_levels(series: pd.Series) -> list:
    """Grup duzeylerini KARARLI bicimde siralar.

    Onceki surumde .unique() kullaniliyordu; bu, gruplarin sirasini veri
    dosyasindaki satir sirasina birakiyordu. Ayni veri farkli siralanmis
    olarak yuklendiginde 1. ve 2. grup yer degistiriyor, dolayisiyla t
    istatistiginin isareti ters donuyordu. Sayisal kodlar sayisal olarak,
    metin kodlar alfabetik olarak siralanir.
    """
    levels = [g for g in pd.unique(series.dropna())]
    numeric = []
    for g in levels:
        try:
            numeric.append(float(g))
        except (TypeError, ValueError):
            numeric = None
            break
    if numeric is not None:
        return [g for _, g in sorted(zip(numeric, levels), key=lambda t: t[0])]
    return sorted(levels, key=lambda g: str(g))


@app.post("/ttest-multiple")
def ttest_multiple(req: MultipleTTestRequest):
    try:
        cols = [req.groupVar] + list(req.depVars)
        bad = _guard(req.data, cols)
        if bad:
            return bad
        df = _frame(req.data, cols)
        bad = _missing(df, [req.groupVar])
        if bad:
            return bad
        df = df.dropna(subset=[req.groupVar])

        groups = _sorted_levels(df[req.groupVar])
        if len(groups) != 2:
            return {"error": f"Grup değişkeninizde tam olarak 2 kategori olmalıdır. "
                             f"Sizde {len(groups)} bulundu."}
        g1_val, g2_val = groups[0], groups[1]
        m1_mask = df[req.groupVar] == g1_val
        m2_mask = df[req.groupVar] == g2_val

        results = []
        skipped = []
        for var in req.depVars:
            if var not in df.columns or var == req.groupVar:
                skipped.append(var)
                continue
            col = pd.to_numeric(df[var], errors="coerce")
            d1 = col[m1_mask].dropna().to_numpy(dtype=float)
            d2 = col[m2_mask].dropna().to_numpy(dtype=float)
            n1, n2 = d1.size, d2.size
            if n1 < 2 or n2 < 2:
                skipped.append(var)
                continue

            m1, m2 = float(d1.mean()), float(d2.mean())
            v1, v2 = float(d1.var(ddof=1)), float(d2.var(ddof=1))
            std1, std2 = math.sqrt(max(v1, 0.0)), math.sqrt(max(v2, 0.0))

            try:
                _, p_lev = stats.levene(d1, d2, center="mean")
                p_lev = _f(p_lev, 1.0)
            except Exception:
                p_lev = 1.0
            is_equal_var = p_lev >= 0.05

            try:
                t_stat, p_val = stats.ttest_ind(d1, d2, equal_var=is_equal_var)
                t_stat, p_val = _f(t_stat), _f(p_val, 1.0)
            except Exception:
                t_stat, p_val = 0.0, 1.0

            # serbestlik derecesi: Welch kullanildiginda n1+n2-2 DEGILDIR
            if is_equal_var:
                dfree = float(n1 + n2 - 2)
            else:
                a, b = v1 / n1, v2 / n2
                den = (a ** 2) / (n1 - 1) + (b ** 2) / (n2 - 1)
                dfree = _f(((a + b) ** 2) / den, float(n1 + n2 - 2)) if den > 0 \
                    else float(n1 + n2 - 2)

            pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
            pooled_sd = math.sqrt(max(pooled_var, 0.0))
            cohens_d = abs(m1 - m2) / pooled_sd if pooled_sd > 0 else 0.0
            # Hedges duzeltmesi (kucuk orneklemde Cohen's d yanlidir)
            J = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0) if (n1 + n2) > 3 else 1.0
            hedges_g = cohens_d * J

            diff = m1 - m2
            se_diff = math.sqrt(v1 / n1 + v2 / n2) if not is_equal_var else \
                (pooled_sd * math.sqrt(1.0 / n1 + 1.0 / n2) if pooled_sd > 0 else 0.0)
            tcrit = float(stats.t.ppf(0.975, dfree)) if dfree > 0 else 0.0

            results.append({
                # --- mevcut alanlar ---
                "varName": var,
                "is_equal_var": bool(is_equal_var),
                "levene_p": p_lev,
                "t": t_stat,
                "p": p_val,
                "cohens_d": _f(cohens_d),
                "g1": {"val": str(g1_val), "n": n1, "mean": m1, "std": std1},
                "g2": {"val": str(g2_val), "n": n2, "mean": m2, "std": std2},
                # --- yeni alanlar ---
                "df": _f(dfree),
                "hedges_g": _f(hedges_g),
                "meanDiff": _f(diff),
                "seDiff": _f(se_diff),
                "ciLow": _f(diff - tcrit * se_diff),
                "ciHigh": _f(diff + tcrit * se_diff),
                "testUsed": "Student" if is_equal_var else "Welch",
            })

        return {
            "groupVar": req.groupVar,
            "originalGroups": [str(g1_val), str(g2_val)],
            "results": results,
            "skipped": skipped,
        }
    except Exception as e:
        log.exception("ttest failed")
        return {"error": f"T-Testi Hatası: {e}"}
    finally:
        gc.collect()


# ==========================================================================
# TEK YONLU ANOVA
# ==========================================================================

def _post_hoc(groups_data, group_stats, labels, method):
    """Ikili karsilastirmalar.

    Onceki surumde yalnizca 'Bonferroni' secildiginde duzeltme uygulaniyor,
    Tukey/Scheffe secildiginde ise DUZELTILMEMIS ikili t-testleri o adla
    raporlaniyordu. Artik her yontem kendi dagilimiyla hesaplanir.
    """
    out, detail = [], []
    kg = len(groups_data)
    ns = np.array([g.size for g in groups_data], dtype=float)
    means = np.array([g.mean() if g.size else np.nan for g in groups_data])
    N = float(ns.sum())
    if kg < 2 or N - kg <= 0:
        return out, detail

    # gruplar arasi ortak varyans (MSE)
    ss_w = float(sum(((g - g.mean()) ** 2).sum() for g in groups_data if g.size > 1))
    df_w = N - kg
    mse = ss_w / df_w if df_w > 0 else 0.0

    pairs = list(itertools.combinations(range(kg), 2))
    ncomp = len(pairs)
    m = str(method or "").strip().lower()

    for i, j in pairs:
        if groups_data[i].size < 2 or groups_data[j].size < 2:
            continue
        diff = float(means[i] - means[j])
        se = math.sqrt(mse * (1.0 / ns[i] + 1.0 / ns[j])) if mse > 0 else 0.0

        if m.startswith("tukey"):
            if se <= 0:
                continue
            q = abs(diff) / (se / math.sqrt(2.0))
            try:
                p_adj = float(stats.studentized_range.sf(q, kg, df_w))
            except Exception:
                p_adj = 1.0
            stat_name, stat_val = "q", q
        elif m.startswith("scheffe") or m.startswith("scheffé"):
            if se <= 0:
                continue
            fstat = (diff ** 2) / (se ** 2) / (kg - 1)
            p_adj = float(stats.f.sf(fstat, kg - 1, df_w))
            stat_name, stat_val = "F", fstat
        else:
            # Bonferroni (varsayilan) ve LSD: ortak varyansli t
            if se <= 0:
                continue
            tval = diff / se
            p_raw = float(2.0 * stats.t.sf(abs(tval), df_w))
            p_adj = min(1.0, p_raw * ncomp) if m.startswith("bonferroni") else p_raw
            stat_name, stat_val = "t", tval

        p_adj = _f(p_adj, 1.0)
        hi, lo = (i, j) if group_stats[i]["mean"] > group_stats[j]["mean"] else (j, i)
        detail.append({"a": str(labels[i]), "b": str(labels[j]),
                       "meanDiff": _f(diff), "se": _f(se),
                       "stat": _f(stat_val), "statName": stat_name,
                       "p": p_adj, "sig": bool(p_adj < 0.05)})
        if p_adj < 0.05:
            out.append({"higher": str(labels[hi]), "lower": str(labels[lo])})
    return out, detail


@app.post("/anova-multiple")
def anova_multiple(req: MultipleANOVARequest):
    try:
        cols = [req.groupVar] + list(req.depVars)
        bad = _guard(req.data, cols)
        if bad:
            return bad
        df = _frame(req.data, cols)
        bad = _missing(df, [req.groupVar])
        if bad:
            return bad
        df = df.dropna(subset=[req.groupVar])

        levels = _sorted_levels(df[req.groupVar])
        if len(levels) < 2:
            return {"error": f"ANOVA için grup değişkeninizde en az 2, tercihen 3 "
                             f"kategori olmalıdır. Sizde {len(levels)} bulundu."}
        masks = [(df[req.groupVar] == g) for g in levels]

        results, skipped = [], []
        for var in req.depVars:
            if var not in df.columns or var == req.groupVar:
                skipped.append(var)
                continue
            col = pd.to_numeric(df[var], errors="coerce")
            gdata = [col[mk].dropna().to_numpy(dtype=float) for mk in masks]
            total = np.concatenate([g for g in gdata if g.size]) if any(
                g.size for g in gdata) else np.array([])
            tot_n = total.size
            if tot_n == 0:
                skipped.append(var)
                continue

            gstats = []
            for g, lab in zip(gdata, levels):
                gstats.append({
                    "val": str(lab), "n": int(g.size),
                    "mean": _f(g.mean()) if g.size else 0.0,
                    "std": _f(g.std(ddof=1)) if g.size > 1 else 0.0,
                })

            usable = [g for g in gdata if g.size >= 2]
            if len(usable) < 2:
                skipped.append(var)
                continue

            try:
                F_stat, p_val = stats.f_oneway(*usable)
                F_stat, p_val = _f(F_stat), _f(p_val, 1.0)
            except Exception:
                F_stat, p_val = 0.0, 1.0

            # --- yeni: kareler toplami, etki buyuklugu, Levene, Welch ---
            kg = len(usable)
            N = float(sum(g.size for g in usable))
            grand = float(np.concatenate(usable).mean())
            ss_b = float(sum(g.size * (g.mean() - grand) ** 2 for g in usable))
            ss_w = float(sum(((g - g.mean()) ** 2).sum() for g in usable))
            ss_t = ss_b + ss_w
            df_b, df_w = kg - 1, N - kg
            ms_w = ss_w / df_w if df_w > 0 else 0.0
            eta2 = ss_b / ss_t if ss_t > 0 else 0.0
            omega2 = ((ss_b - df_b * ms_w) / (ss_t + ms_w)) if (ss_t + ms_w) > 0 else 0.0

            try:
                _, lev_p = stats.levene(*usable, center="mean")
                lev_p = _f(lev_p, 1.0)
            except Exception:
                lev_p = 1.0

            welch = None
            try:                                     # Welch'in duzeltilmis F'i
                w = np.array([g.size / g.var(ddof=1) for g in usable
                              if g.var(ddof=1) > 0], dtype=float)
                if w.size == kg:
                    mu = np.array([g.mean() for g in usable], dtype=float)
                    sw = w.sum()
                    mbar = float((w * mu).sum() / sw)
                    num = float((w * (mu - mbar) ** 2).sum()) / (kg - 1)
                    lam = float(sum((1 - wi / sw) ** 2 / (g.size - 1)
                                    for wi, g in zip(w, usable)))
                    lam = 3.0 * lam / (kg ** 2 - 1)
                    Fw = num / (1.0 + 2.0 * (kg - 2) / (kg + 1) * lam)
                    df2 = 1.0 / (3.0 * lam / (kg ** 2 - 1)) if lam > 0 else df_w
                    welch = {"F": _f(Fw), "df1": float(kg - 1), "df2": _f(df2),
                             "p": _f(stats.f.sf(Fw, kg - 1, df2), 1.0)}
            except Exception:
                welch = None

            pairs, detail = ([], [])
            if p_val < 0.05:
                pairs, detail = _post_hoc(gdata, gstats, levels, req.postHoc)

            results.append({
                # --- mevcut alanlar ---
                "varName": var,
                "F": F_stat,
                "p": p_val,
                "groups": gstats,
                "total": {"n": int(tot_n), "mean": _f(total.mean()),
                          "std": _f(total.std(ddof=1)) if tot_n > 1 else 0.0},
                "postHocPairs": pairs,
                # --- yeni alanlar ---
                "df_between": float(df_b), "df_within": float(df_w),
                "ss_between": _f(ss_b), "ss_within": _f(ss_w), "ss_total": _f(ss_t),
                "ms_between": _f(ss_b / df_b) if df_b > 0 else 0.0,
                "ms_within": _f(ms_w),
                "eta2": _f(eta2), "omega2": _f(omega2),
                "levene_p": lev_p,
                "welch": welch,
                "postHocDetail": detail,
            })

        return {
            "groupVar": req.groupVar,
            "postHoc": req.postHoc,
            "originalGroups": [str(g) for g in levels],
            "results": results,
            "skipped": skipped,
        }
    except Exception as e:
        log.exception("anova failed")
        return {"error": f"ANOVA Hatası: {e}"}
    finally:
        gc.collect()


# ==========================================================================
# NON-PARAMETRIK TESTLER
# --------------------------------------------------------------------------
# Bagimsiz orneklem t-testinin karsiligi : Mann-Whitney U
# Tek yonlu ANOVA'nin karsiligi          : Kruskal-Wallis H (+ Dunn post-hoc)
#
# Her ikisi de baglanmis siralar (ties) icin duzeltme icerir; bu duzeltme
# yapilmazsa baglanma cok olan olcek verilerinde p degeri oldugundan buyuk
# cikar (testi muhafazakar yapar).
# ==========================================================================

def _ranks_with_ties(x: np.ndarray):
    """Ortalama siralar ve baglanma duzeltme terimi Sum(t^3 - t)."""
    r = stats.rankdata(x)                      # baglanmalarda ortalama sira
    _, counts = np.unique(x, return_counts=True)
    tie_sum = float(((counts ** 3) - counts).sum())
    return r, tie_sum


def _grup_ozeti(vals: np.ndarray, ranks: np.ndarray, label) -> dict:
    n = int(vals.size)
    return {
        "val": str(label), "n": n,
        "mean": _f(vals.mean()) if n else 0.0,
        "std": _f(vals.std(ddof=1)) if n > 1 else 0.0,
        "median": _f(np.median(vals)) if n else 0.0,
        "meanRank": _f(ranks.mean()) if n else 0.0,
        "sumRank": _f(ranks.sum()) if n else 0.0,
    }


# --------------------------------------------------------------------------
# Mann-Whitney U
# --------------------------------------------------------------------------

@app.post("/mannwhitney-multiple")
def mannwhitney_multiple(req: MultipleTTestRequest):
    try:
        cols = [req.groupVar] + list(req.depVars)
        bad = _guard(req.data, cols)
        if bad:
            return bad
        df = _frame(req.data, cols)
        bad = _missing(df, [req.groupVar])
        if bad:
            return bad
        df = df.dropna(subset=[req.groupVar])

        groups = _sorted_levels(df[req.groupVar])
        if len(groups) != 2:
            return {"error": f"Mann-Whitney U testi için grup değişkeninizde tam olarak "
                             f"2 kategori olmalıdır. Sizde {len(groups)} bulundu."}
        g1_val, g2_val = groups[0], groups[1]
        m1 = df[req.groupVar] == g1_val
        m2 = df[req.groupVar] == g2_val

        results, skipped = [], []
        for var in req.depVars:
            if var not in df.columns or var == req.groupVar:
                skipped.append(var)
                continue
            col = pd.to_numeric(df[var], errors="coerce")
            d1 = col[m1].dropna().to_numpy(dtype=float)
            d2 = col[m2].dropna().to_numpy(dtype=float)
            n1, n2 = d1.size, d2.size
            if n1 < 2 or n2 < 2:
                skipped.append(var)
                continue

            hepsi = np.concatenate([d1, d2])
            N = n1 + n2
            r, tie_sum = _ranks_with_ties(hepsi)
            r1, r2 = r[:n1], r[n1:]

            R1 = float(r1.sum())
            U1 = R1 - n1 * (n1 + 1) / 2.0
            U2 = n1 * n2 - U1
            U = min(U1, U2)

            # baglanma duzeltmeli standart sapma ve surekli duzeltmeli z
            mu = n1 * n2 / 2.0
            var_u = (n1 * n2 / 12.0) * ((N + 1) - tie_sum / (N * (N - 1.0))) \
                if N > 1 else 0.0
            sd_u = math.sqrt(var_u) if var_u > 0 else 0.0
            if sd_u > 0:
                fark = U1 - mu
                duz = math.copysign(0.5, fark) if fark != 0 else 0.0
                z = (fark - duz) / sd_u
            else:
                z = 0.0

            # p degeri: baglanma yoksa ve orneklem kucukse tam (exact) dagilim
            try:
                mw = stats.mannwhitneyu(
                    d1, d2, alternative="two-sided",
                    method=("exact" if (tie_sum == 0 and n1 <= 20 and n2 <= 20)
                            else "asymptotic"))
                p = _f(mw.pvalue, 1.0)
                yontem = ("Tam (exact) dağılım" if (tie_sum == 0 and n1 <= 20 and n2 <= 20)
                          else "Normal yaklaşım (bağlanma düzeltmeli)")
            except Exception:
                p = _f(2.0 * stats.norm.sf(abs(z)), 1.0)
                yontem = "Normal yaklaşım (bağlanma düzeltmeli)"

            results.append({
                "varName": var,
                "U": _f(U), "U1": _f(U1), "U2": _f(U2),
                "W": _f(R1),                       # birinci grubun sıra toplamı
                "z": _f(z), "p": p,
                "r": _f(abs(z) / math.sqrt(N)) if N > 0 else 0.0,   # etki büyüklüğü
                "n": int(N), "ties": tie_sum > 0, "method": yontem,
                "g1": _grup_ozeti(d1, r1, g1_val),
                "g2": _grup_ozeti(d2, r2, g2_val),
            })

        return {"groupVar": req.groupVar, "test": "Mann-Whitney U",
                "originalGroups": [str(g1_val), str(g2_val)],
                "results": results, "skipped": skipped}
    except Exception as e:
        log.exception("mannwhitney failed")
        return {"error": f"Mann-Whitney U Testi Hatası: {e}"}
    finally:
        gc.collect()


# --------------------------------------------------------------------------
# Kruskal-Wallis H + Dunn post-hoc
# --------------------------------------------------------------------------

def _dunn(gruplar, labels, ranks_by_group, N, tie_sum):
    """Dunn (1964) ikili karsilastirmalari, Bonferroni duzeltmeli."""
    k = len(gruplar)
    pairs = list(itertools.combinations(range(k), 2))
    ncomp = len(pairs)
    # ortak standart hata terimi
    taban = (N * (N + 1) / 12.0) - (tie_sum / (12.0 * (N - 1.0))) if N > 1 else 0.0
    out, detay = [], []
    for i, j in pairs:
        ni, nj = gruplar[i].size, gruplar[j].size
        if ni < 1 or nj < 1:
            continue
        ri = float(ranks_by_group[i].mean())
        rj = float(ranks_by_group[j].mean())
        se = math.sqrt(taban * (1.0 / ni + 1.0 / nj)) if taban > 0 else 0.0
        if se <= 0:
            continue
        z = (ri - rj) / se
        p_raw = 2.0 * float(stats.norm.sf(abs(z)))
        p_adj = min(1.0, p_raw * ncomp)
        sig = p_adj < 0.05
        hi, lo = (i, j) if ri > rj else (j, i)
        detay.append({"a": str(labels[i]), "b": str(labels[j]),
                      "meanRankDiff": _f(ri - rj), "se": _f(se),
                      "stat": _f(z), "statName": "z",
                      "pRaw": _f(p_raw, 1.0), "p": _f(p_adj, 1.0), "sig": bool(sig)})
        if sig:
            out.append({"higher": str(labels[hi]), "lower": str(labels[lo])})
    return out, detay


@app.post("/kruskal-multiple")
def kruskal_multiple(req: MultipleANOVARequest):
    try:
        cols = [req.groupVar] + list(req.depVars)
        bad = _guard(req.data, cols)
        if bad:
            return bad
        df = _frame(req.data, cols)
        bad = _missing(df, [req.groupVar])
        if bad:
            return bad
        df = df.dropna(subset=[req.groupVar])

        levels = _sorted_levels(df[req.groupVar])
        if len(levels) < 2:
            return {"error": f"Kruskal-Wallis testi için grup değişkeninizde en az 2, "
                             f"tercihen 3 kategori olmalıdır. Sizde {len(levels)} bulundu."}
        masks = [(df[req.groupVar] == g) for g in levels]

        results, skipped = [], []
        for var in req.depVars:
            if var not in df.columns or var == req.groupVar:
                skipped.append(var)
                continue
            col = pd.to_numeric(df[var], errors="coerce")
            gdata = [col[mk].dropna().to_numpy(dtype=float) for mk in masks]
            kullanilabilir = [(g, lab) for g, lab in zip(gdata, levels) if g.size >= 2]
            if len(kullanilabilir) < 2:
                skipped.append(var)
                continue
            gd = [g for g, _ in kullanilabilir]
            lb = [lab for _, lab in kullanilabilir]

            hepsi = np.concatenate(gd)
            N = int(hepsi.size)
            r, tie_sum = _ranks_with_ties(hepsi)
            # her grubun siralarini ayir
            ranks_by_group, bas = [], 0
            for g in gd:
                ranks_by_group.append(r[bas:bas + g.size])
                bas += g.size

            try:
                kw = stats.kruskal(*gd)
                H, p = _f(kw.statistic), _f(kw.pvalue, 1.0)
            except Exception:
                H, p = 0.0, 1.0
            k = len(gd)
            dfree = k - 1

            gstats = [_grup_ozeti(g, rk, lab)
                      for g, rk, lab in zip(gd, ranks_by_group, lb)]
            # analiz disi kalan gruplar da tabloda gorunsun
            for g, lab in zip(gdata, levels):
                if lab not in lb:
                    gstats.append({"val": str(lab), "n": int(g.size),
                                   "mean": _f(g.mean()) if g.size else 0.0,
                                   "std": _f(g.std(ddof=1)) if g.size > 1 else 0.0,
                                   "median": _f(np.median(g)) if g.size else 0.0,
                                   "meanRank": None, "sumRank": None})

            # etki buyuklukleri
            eps2 = _f(H / (N - 1.0)) if N > 1 else 0.0
            eta2 = _f((H - k + 1) / (N - k)) if N > k else 0.0

            pairs, detay = ([], [])
            if p < 0.05 and k >= 2:
                pairs, detay = _dunn(gd, lb, ranks_by_group, N, tie_sum)

            results.append({
                "varName": var, "H": H, "df": int(dfree), "p": p, "n": N,
                "epsilon2": eps2, "eta2H": eta2, "ties": tie_sum > 0,
                "groups": gstats, "postHocPairs": pairs, "postHocDetail": detay,
                "total": {"n": N, "mean": _f(hepsi.mean()),
                          "std": _f(hepsi.std(ddof=1)) if N > 1 else 0.0,
                          "median": _f(np.median(hepsi))},
            })

        return {"groupVar": req.groupVar, "test": "Kruskal-Wallis H",
                "postHoc": "Dunn", "originalGroups": [str(g) for g in levels],
                "results": results, "skipped": skipped}
    except Exception as e:
        log.exception("kruskal failed")
        return {"error": f"Kruskal-Wallis Testi Hatası: {e}"}
    finally:
        gc.collect()


# ==========================================================================
# ESLESTIRILMIS (BAGIMLI) ORNEKLEM TESTLERI
# --------------------------------------------------------------------------
# Ayni kisilerden iki kez olcum alindiginda (ontest-sontest, oncesi-sonrasi)
# kullanilir:
#   Parametrik      : Bagimli Orneklem t-Testi
#   Non-parametrik  : Wilcoxon Isaretli Siralar Testi
#
# Fark her iki testte de d = (2. olcum) - (1. olcum) olarak hesaplanir; boylece
# pozitif deger "ikinci olcum daha yuksek" anlamina gelir ve iki testin yonu
# birbiriyle tutarli kalir. Bu, raporlarda acikca belirtilir.
# ==========================================================================

class PairedRequest(BaseModel):
    pairs: List[Dict[str, Any]]        # [{"a": "Ontest", "b": "Sontest", "label": "..."}]
    data: List[Dict[str, Any]]


def _pair_cols(pairs) -> List[str]:
    out = []
    for p in pairs or []:
        for k in ("a", "b"):
            v = p.get(k)
            if isinstance(v, str) and v and v not in out:
                out.append(v)
    return out


def _pair_prep(df, p):
    """Bir cift icin eslesmis (listwise) gozlemleri dondurur."""
    a, b = p.get("a"), p.get("b")
    if not a or not b or a == b:
        return None
    if a not in df.columns or b not in df.columns:
        return None
    x = pd.to_numeric(df[a], errors="coerce")
    y = pd.to_numeric(df[b], errors="coerce")
    ok = x.notna() & y.notna()          # eslesmis veri: iki olcum de dolu olmali
    xa = x[ok].to_numpy(dtype=float)
    yb = y[ok].to_numpy(dtype=float)
    if xa.size < 3:
        return None
    etiket = p.get("label") or f"{a} - {b}"
    return a, b, xa, yb, str(etiket)


# --------------------------------------------------------------------------
# Bagimli orneklem t-testi
# --------------------------------------------------------------------------

@app.post("/pairedttest-multiple")
def paired_ttest_multiple(req: PairedRequest):
    try:
        cols = _pair_cols(req.pairs)
        bad = _guard(req.data, cols)
        if bad:
            return bad
        if not cols:
            return {"error": "Karşılaştırılacak ölçüm çifti belirtilmedi."}
        df = _frame(req.data, cols)
        bad = _missing(df, cols)
        if bad:
            return bad

        results, skipped = [], []
        for p in req.pairs:
            hazir = _pair_prep(df, p)
            if hazir is None:
                skipped.append(p.get("label") or f"{p.get('a')} - {p.get('b')}")
                continue
            a, b, xa, yb, etiket = hazir
            n = int(xa.size)
            d = yb - xa                        # 2. olcum - 1. olcum
            ort_d = float(d.mean())
            sd_d = float(d.std(ddof=1)) if n > 1 else 0.0
            se_d = sd_d / math.sqrt(n) if n > 0 and sd_d > 0 else 0.0

            try:
                tt = stats.ttest_rel(yb, xa)
                t_stat, p_val = _f(tt.statistic), _f(tt.pvalue, 1.0)
            except Exception:
                t_stat, p_val = 0.0, 1.0
            dfree = n - 1

            # olcumler arasi korelasyon (SPSS de raporlar)
            try:
                if xa.std() > 0 and yb.std() > 0:
                    pr = stats.pearsonr(xa, yb)
                    r_corr, r_p = _f(pr.statistic), _f(pr.pvalue, 1.0)
                else:
                    r_corr, r_p = 0.0, 1.0
            except Exception:
                r_corr, r_p = 0.0, 1.0

            # etki buyuklugu: Cohen's d_z (farklarin standart sapmasi uzerinden)
            dz = ort_d / sd_d if sd_d > 0 else 0.0
            # ortalama standart sapma uzerinden d_av (bazi kaynaklar bunu ister)
            sd_ort = (float(xa.std(ddof=1)) + float(yb.std(ddof=1))) / 2.0 if n > 1 else 0.0
            dav = ort_d / sd_ort if sd_ort > 0 else 0.0

            tcrit = float(stats.t.ppf(0.975, dfree)) if dfree > 0 else 0.0
            results.append({
                "label": etiket, "a": a, "b": b, "n": n,
                "aMean": _f(xa.mean()), "aStd": _f(xa.std(ddof=1)) if n > 1 else 0.0,
                "bMean": _f(yb.mean()), "bStd": _f(yb.std(ddof=1)) if n > 1 else 0.0,
                "meanDiff": _f(ort_d), "sdDiff": _f(sd_d), "seDiff": _f(se_d),
                "ciLow": _f(ort_d - tcrit * se_d), "ciHigh": _f(ort_d + tcrit * se_d),
                "t": t_stat, "df": int(dfree), "p": p_val,
                "cohens_d": _f(dz), "cohens_dav": _f(dav),
                "corr": r_corr, "corrP": r_p,
            })

        return {"test": "Bağımlı Örneklem t-Testi", "results": results, "skipped": skipped}
    except Exception as e:
        log.exception("paired ttest failed")
        return {"error": f"Bağımlı Örneklem t-Testi Hatası: {e}"}
    finally:
        gc.collect()


# --------------------------------------------------------------------------
# Wilcoxon isaretli siralar testi
# --------------------------------------------------------------------------

@app.post("/wilcoxon-multiple")
def wilcoxon_multiple(req: PairedRequest):
    try:
        cols = _pair_cols(req.pairs)
        bad = _guard(req.data, cols)
        if bad:
            return bad
        if not cols:
            return {"error": "Karşılaştırılacak ölçüm çifti belirtilmedi."}
        df = _frame(req.data, cols)
        bad = _missing(df, cols)
        if bad:
            return bad

        results, skipped = [], []
        for p in req.pairs:
            hazir = _pair_prep(df, p)
            if hazir is None:
                skipped.append(p.get("label") or f"{p.get('a')} - {p.get('b')}")
                continue
            a, b, xa, yb, etiket = hazir
            n_toplam = int(xa.size)
            d = yb - xa                        # 2. olcum - 1. olcum

            esit = int((d == 0).sum())         # Wilcoxon'da sifir farklar atilir
            dn = d[d != 0]
            n = int(dn.size)
            if n < 3:
                skipped.append(etiket)
                continue

            mutlak = np.abs(dn)
            r, tie_sum = _ranks_with_ties(mutlak)
            poz = r[dn > 0]
            neg = r[dn < 0]
            W_poz = float(poz.sum())
            W_neg = float(neg.sum())
            T = min(W_poz, W_neg)

            # baglanma duzeltmeli varyans ve surekli duzeltmeli z
            mu = n * (n + 1) / 4.0
            var_t = (n * (n + 1) * (2 * n + 1)) / 24.0 - tie_sum / 48.0
            sd_t = math.sqrt(var_t) if var_t > 0 else 0.0
            if sd_t > 0:
                fark = W_poz - mu
                duz = math.copysign(0.5, fark) if fark != 0 else 0.0
                z = (fark - duz) / sd_t
            else:
                z = 0.0

            try:
                tam = (tie_sum == 0 and esit == 0 and n <= 25)
                wt = stats.wilcoxon(yb, xa, alternative="two-sided",
                                    zero_method="wilcox",
                                    method=("exact" if tam else "approx"))
                p_val = _f(wt.pvalue, 1.0)
                yontem = ("Tam (exact) dağılım" if tam
                          else "Normal yaklaşım (bağlanma düzeltmeli)")
            except Exception:
                p_val = _f(2.0 * stats.norm.sf(abs(z)), 1.0)
                yontem = "Normal yaklaşım (bağlanma düzeltmeli)"

            results.append({
                "label": etiket, "a": a, "b": b,
                "n": n_toplam, "nUsed": n,
                "aMean": _f(xa.mean()), "aStd": _f(xa.std(ddof=1)) if n_toplam > 1 else 0.0,
                "aMedian": _f(np.median(xa)),
                "bMean": _f(yb.mean()), "bStd": _f(yb.std(ddof=1)) if n_toplam > 1 else 0.0,
                "bMedian": _f(np.median(yb)),
                "neg": {"n": int(neg.size),
                        "meanRank": _f(neg.mean()) if neg.size else 0.0,
                        "sumRank": _f(W_neg)},
                "pos": {"n": int(poz.size),
                        "meanRank": _f(poz.mean()) if poz.size else 0.0,
                        "sumRank": _f(W_poz)},
                "esit": esit,
                "T": _f(T), "z": _f(z), "p": p_val,
                "r": _f(abs(z) / math.sqrt(n_toplam)) if n_toplam > 0 else 0.0,
                "medianDiff": _f(np.median(d)),
                "ties": tie_sum > 0, "method": yontem,
            })

        return {"test": "Wilcoxon İşaretli Sıralar", "results": results, "skipped": skipped}
    except Exception as e:
        log.exception("wilcoxon failed")
        return {"error": f"Wilcoxon İşaretli Sıralar Testi Hatası: {e}"}
    finally:
        gc.collect()


# ==========================================================================
# KORELASYON ANALIZI
# --------------------------------------------------------------------------
# Iki surekli degisken arasindaki dogrusal iliski:
#   Parametrik      : Pearson korelasyon katsayisi (r)
#   Non-parametrik  : Spearman sira korelasyonu (rho) — normallik ve
#                     dogrusallik varsayimi gerektirmez, sirali (ordinal)
#                     olceklerde ve uc degerli verilerde tercih edilir.
#
# Eksik veri: IKILI (pairwise) silme. Her degisken cifti icin yalnizca o
# ciftte iki degeri de dolu olan gozlemler kullanilir; bu, SPSS'in varsayilan
# davranisidir ve tek bir degiskendeki bosluklarin tum matrisi kucultmesini
# onler. Bunun bedeli, hucrelerin farkli n degerlerine dayanmasidir; bu yuzden
# n araligi yanitta ayrica dondurulur.
# ==========================================================================

MAX_CORR_VARS = int(os.getenv("AIS_MAX_CORR_VARS", "40"))


class CorrelationRequest(BaseModel):
    vars: List[str]
    data: List[Dict[str, Any]]
    method: Optional[str] = "pearson"       # "pearson" | "spearman"


def _fisher_ci(r: float, n: int):
    """Fisher z donusumu ile r icin %95 guven araligi."""
    if n < 4 or not math.isfinite(r) or abs(r) >= 1.0:
        return None, None
    z = 0.5 * math.log((1.0 + r) / (1.0 - r))
    se = 1.0 / math.sqrt(n - 3)
    lo, hi = z - 1.959963984540054 * se, z + 1.959963984540054 * se
    return math.tanh(lo), math.tanh(hi)


@app.post("/correlation-multiple")
def correlation_multiple(req: CorrelationRequest):
    try:
        secilen = [v for v in dict.fromkeys(req.vars or [])]   # sirayi koru
        bad = _guard(req.data, secilen)
        if bad:
            return bad
        if len(secilen) < 2:
            return {"error": "Korelasyon analizi için en az iki değişken seçmelisiniz."}
        if len(secilen) > MAX_CORR_VARS:
            return {"error": f"Korelasyon matrisi en fazla {MAX_CORR_VARS} değişkenle "
                             f"oluşturulabilir. Siz {len(secilen)} değişken seçtiniz."}

        df = _frame(req.data, secilen)
        bad = _missing(df, secilen)
        if bad:
            return bad

        yontem = (req.method or "pearson").strip().lower()
        if yontem not in ("pearson", "spearman"):
            yontem = "pearson"

        # Sayisala cevir; hicbir gecerli degeri olmayan veya SABIT olan
        # degiskenler analiz disi birakilir (sabit degiskenle r tanimsizdir).
        kolonlar, skipped = [], []
        seriler = {}
        for v in secilen:
            s = pd.to_numeric(df[v], errors="coerce")
            gecerli = s.dropna()
            if gecerli.size < 3 or float(gecerli.std(ddof=1) or 0.0) <= 0:
                skipped.append(v)
                continue
            kolonlar.append(v)
            seriler[v] = s

        k = len(kolonlar)
        if k < 2:
            return {"error": "Korelasyon için en az iki geçerli (sabit olmayan, "
                             "en az 3 gözlemli) sayısal değişken gereklidir."}

        betimsel = []
        for v in kolonlar:
            g = seriler[v].dropna().to_numpy(dtype=float)
            betimsel.append({
                "varName": v, "n": int(g.size),
                "mean": _f(g.mean()), "std": _f(g.std(ddof=1)) if g.size > 1 else 0.0,
                "min": _f(g.min()), "max": _f(g.max()),
                "skewness": _f(stats.skew(g, bias=False)) if g.size > 3 else 0.0,
                "kurtosis": _f(stats.kurtosis(g, bias=False)) if g.size > 4 else 0.0,
            })

        # --- ikili (pairwise) korelasyon matrisi ---
        matris = [[None] * k for _ in range(k)]
        n_min, n_max = None, None
        for i in range(k):
            matris[i][i] = {"r": 1.0, "p": 0.0, "n": int(seriler[kolonlar[i]].notna().sum()),
                            "ciLow": None, "ciHigh": None, "diag": True}
            for j in range(i + 1, k):
                a = seriler[kolonlar[i]]
                b = seriler[kolonlar[j]]
                ok = a.notna() & b.notna()
                xa = a[ok].to_numpy(dtype=float)
                yb = b[ok].to_numpy(dtype=float)
                n = int(xa.size)
                if n < 3 or xa.std() == 0 or yb.std() == 0:
                    hucre = {"r": None, "p": None, "n": n,
                             "ciLow": None, "ciHigh": None}
                else:
                    try:
                        if yontem == "spearman":
                            res = stats.spearmanr(xa, yb)
                            r_val, p_val = float(res.statistic), float(res.pvalue)
                        else:
                            res = stats.pearsonr(xa, yb)
                            r_val, p_val = float(res.statistic), float(res.pvalue)
                    except Exception:
                        r_val, p_val = float("nan"), float("nan")
                    if not math.isfinite(r_val):
                        hucre = {"r": None, "p": None, "n": n,
                                 "ciLow": None, "ciHigh": None}
                    else:
                        lo, hi = _fisher_ci(r_val, n)
                        hucre = {"r": _f(r_val), "p": _f(p_val, 1.0), "n": n,
                                 "ciLow": _fo(lo), "ciHigh": _fo(hi)}
                    n_min = n if n_min is None else min(n_min, n)
                    n_max = n if n_max is None else max(n_max, n)
                matris[i][j] = hucre
                matris[j][i] = dict(hucre)

        return {
            "test": ("Spearman Sıra Korelasyonu" if yontem == "spearman"
                     else "Pearson Korelasyon"),
            "method": yontem,
            "statLabel": "rs" if yontem == "spearman" else "r",
            "vars": kolonlar,
            "matrix": matris,
            "descriptives": betimsel,
            "nMin": n_min, "nMax": n_max,
            "deletion": "pairwise",
            "skipped": skipped,
        }
    except Exception as e:
        log.exception("correlation failed")
        return {"error": f"Korelasyon Analizi Hatası: {e}"}
    finally:
        gc.collect()


# ==========================================================================
# ROC EGRISI ANALIZI
# --------------------------------------------------------------------------
# Tanisal dogruluk calismalarinin standart araci. Surekli bir belirtecin
# (marker/test) ikili bir durumu (hasta / saglikli) ne kadar iyi ayirt
# ettigini olcer.
#
# Egri altinda kalan alan (AUC) ve standart hatasi DeLong, DeLong ve
# Clarke-Pearson (1988) yontemiyle hesaplanir; Sun ve Xu (2014) hizli
# algoritmasi kullanilir. Bu yontem hem tek bir AUC'nin guven araligini hem
# de AYNI hastalarda olculen iki belirtecin AUC farkinin testini ayni
# kovaryans yapisindan uretir; eslesmis (paired) karsilastirma icin dogru
# olan budur.
#
# Optimal kesme noktasi Youden indeksi (J = duyarlilik + ozgulluk - 1) ile
# secilir. Oranlarin guven araliklari Wilson skor yontemiyle, olabilirlik
# oranlarininki Simel ve ark. (1991) log yontemiyle hesaplanir.
# ==========================================================================

MAX_ROC_MARKERS = int(os.getenv("AIS_MAX_ROC_MARKERS", "10"))
MAX_ROC_POINTS = 400          # egri cizimi icin dondurulen nokta sayisi tavani


class ROCRequest(BaseModel):
    statusVar: str
    positive: Any                              # pozitif (hasta) duzeyin degeri
    markers: List[Dict[str, Any]]              # [{"name":..., "direction":"higher"|"lower"}]
    data: List[Dict[str, Any]]
    prevalence: Optional[float] = None         # 0-1; verilirse PPD/NPD yeniden hesaplanir


def _wilson(k: float, n: float):
    """Bir oran icin Wilson skor guven araligi (%95)."""
    if n <= 0:
        return None, None
    z = 1.959963984540054
    p = k / n
    den = 1.0 + z * z / n
    orta = (p + z * z / (2.0 * n)) / den
    yari = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return max(0.0, orta - yari), min(1.0, orta + yari)


def _midrank(x: np.ndarray) -> np.ndarray:
    """Baglanmalarda ortalama sira (DeLong algoritmasinin cekirdegi)."""
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = Z.size
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _delong(pos: np.ndarray, neg: np.ndarray):
    """AUC vektoru ve kovaryans matrisi.

    pos : (k, m) pozitif (hasta) olgularin k belirtecteki skorlari
    neg : (k, n) negatif olgularin skorlari
    Ayni olgular uzerinde calisildigi icin dondurulen kovaryans, eslesmis
    AUC farkinin varyansini da verir.
    """
    k, m = pos.shape
    n = neg.shape[1]
    tx = np.vstack([_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1.0) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n                 # hasta olgularin katkisi
    v10 = 1.0 - (tz[:, m:] - ty) / m           # saglikli olgularin katkisi
    sx = np.cov(v01, ddof=1).reshape(k, k)
    sy = np.cov(v10, ddof=1).reshape(k, k)
    S = sx / m + sy / n
    return aucs, S


def _roc_egrisi(skor: np.ndarray, etiket: np.ndarray):
    """Butun esiklerde (1-ozgulluk, duyarlilik) ciftlerini uretir.

    Kural: skor >= esik ise test POZITIF. Baglanmalarda ayni degere sahip
    gozlemler birlikte degerlendirilir; aksi halde egri, aslinda ayirt
    edilemeyen noktalarda merdiven yaparak AUC'yi oldugundan iyi gosterir.
    """
    sira = np.argsort(-skor, kind="mergesort")
    s = skor[sira]
    y = etiket[sira]
    # ayni skor degerinin son indeksi
    farkli = np.where(np.diff(s))[0]
    kesim = np.r_[farkli, s.size - 1]
    tp = np.cumsum(y)[kesim]
    fp = (kesim + 1) - tp
    P = float(y.sum())
    N = float(y.size - P)
    tpr = np.r_[0.0, tp / P] if P > 0 else np.zeros(kesim.size + 1)
    fpr = np.r_[0.0, fp / N] if N > 0 else np.zeros(kesim.size + 1)
    esik = np.r_[np.inf, s[kesim]]
    return fpr, tpr, esik, np.r_[0.0, tp], np.r_[0.0, fp], P, N


@app.post("/roc-multiple")
def roc_multiple(req: ROCRequest):
    try:
        adlar, etiketler = [], {}
        for m in (req.markers or []):
            ad = m.get("name")
            if isinstance(ad, str) and ad and ad not in adlar:
                adlar.append(ad)
                etiketler[ad] = m
        cols = [req.statusVar] + adlar
        bad = _guard(req.data, cols)
        if bad:
            return bad
        if not adlar:
            return {"error": "En az bir test (belirteç) değişkeni seçmelisiniz."}
        if len(adlar) > MAX_ROC_MARKERS:
            return {"error": f"Aynı anda en fazla {MAX_ROC_MARKERS} belirteç "
                             f"analiz edilebilir. Siz {len(adlar)} seçtiniz."}
        if req.statusVar in adlar:
            return {"error": "Durum (altın standart) değişkeni aynı zamanda "
                             "test değişkeni olamaz."}

        df = _frame(req.data, cols)
        bad = _missing(df, cols)
        if bad:
            return bad
        df = df.dropna(subset=[req.statusVar])

        duzeyler = _sorted_levels(df[req.statusVar])
        if len(duzeyler) != 2:
            return {"error": f"Durum değişkeninizde tam olarak 2 kategori olmalıdır "
                             f"(hasta / sağlıklı). Sizde {len(duzeyler)} bulundu."}
        poz_deger = req.positive
        eslesen = [g for g in duzeyler if str(g) == str(poz_deger)]
        if not eslesen:
            return {"error": "Pozitif (hasta) olarak belirttiğiniz değer, durum "
                             "değişkeninde bulunamadı."}
        poz_deger = eslesen[0]
        neg_deger = [g for g in duzeyler if str(g) != str(poz_deger)][0]
        y_tum = (df[req.statusVar].astype(str) == str(poz_deger)).to_numpy()

        prevalans = None
        if req.prevalence is not None:
            try:
                pv = float(req.prevalence)
                if 0.0 < pv < 1.0:
                    prevalans = pv
            except (TypeError, ValueError):
                prevalans = None

        sonuclar, skipped = [], []
        skorlar = {}                   # DeLong karsilastirmasi icin saklanir
        for ad in adlar:
            yon = str(etiketler[ad].get("direction") or "higher").lower()
            ham = pd.to_numeric(df[ad], errors="coerce")
            ok = ham.notna().to_numpy()
            x = ham[ok].to_numpy(dtype=float)
            y = y_tum[ok]
            n_poz = int(y.sum())
            n_neg = int(y.size - n_poz)
            if n_poz < 3 or n_neg < 3 or np.unique(x).size < 2:
                skipped.append(ad)
                continue

            # "Dusuk deger hastalik" secildiyse skoru ters cevir; boylece
            # butun hesaplar tek bir kuralla ("skor >= esik -> pozitif")
            # yurur, esik degeri rapor edilirken orijinal olcege donulur.
            ters = (yon == "lower")
            skor = -x if ters else x
            skorlar[ad] = (skor, y, ok)

            fpr, tpr, esik, tp, fp, P, N = _roc_egrisi(skor, y.astype(float))
            auc_v, S = _delong(skor[y].reshape(1, -1), skor[~y].reshape(1, -1))
            auc = float(auc_v[0])
            se = float(math.sqrt(max(S[0, 0], 0.0)))   # GUVEN ARALIGI icin

            # H0: AUC = 0,5 SINAMASI
            # ------------------------------------------------------------
            # DeLong standart hatasi guven araligi icin dogrudur ama sifir
            # hipotezi ALTINDAKI standart hata degildir; mukemmel ayrimda
            # (AUC = 1) sifira indigi icin test yapilamaz hale gelir. Sinama
            # bu yuzden AUC = 0,5'in tam esdegeri olan Mann-Whitney U
            # dagilimindan, baglanma duzeltmeli sifir varyansiyla yapilir.
            _, tie_sum = _ranks_with_ties(skor)
            NN = float(P_n := (n_poz + n_neg))
            var0 = (n_poz * n_neg / 12.0) * ((NN + 1.0) - tie_sum / (NN * (NN - 1.0))) \
                if NN > 1 else 0.0
            se0 = math.sqrt(max(var0, 0.0)) / (n_poz * n_neg) if var0 > 0 else 0.0
            z_auc = (auc - 0.5) / se0 if se0 > 0 else 0.0
            p_auc = _f(2.0 * stats.norm.sf(abs(z_auc)), 1.0)

            # ---- Youden indeksiyle optimal kesme noktasi ----
            J = tpr - fpr
            i_opt = int(np.argmax(J))
            if i_opt == 0 and J.size > 1:          # (0,0) noktasi secilmesin
                i_opt = int(np.argmax(J[1:])) + 1
            duy = float(tpr[i_opt])
            ozg = float(1.0 - fpr[i_opt])
            GP = float(tp[i_opt])                  # gercek pozitif
            YP = float(fp[i_opt])                  # yanlis pozitif
            YN = P - GP
            GN = N - YP
            kesme_skor = float(esik[i_opt])
            kesme = -kesme_skor if ters else kesme_skor
            if not math.isfinite(kesme):
                kesme = float(x.max()) if not ters else float(x.min())

            dogruluk = (GP + GN) / (P + N) if (P + N) > 0 else 0.0
            ppd = GP / (GP + YP) if (GP + YP) > 0 else None
            npd = GN / (GN + YN) if (GN + YN) > 0 else None

            # olabilirlik oranlari + Simel log guven araligi
            lr_arti = lr_eksi = None
            lr_arti_ci = lr_eksi_ci = (None, None)
            if ozg < 1.0 and duy > 0:
                lr_arti = duy / (1.0 - ozg)
                sh = math.sqrt((1 - duy) / (duy * P) + ozg / ((1 - ozg) * N))
                lr_arti_ci = (math.exp(math.log(lr_arti) - 1.96 * sh),
                              math.exp(math.log(lr_arti) + 1.96 * sh))
            if ozg > 0 and duy < 1.0:
                lr_eksi = (1.0 - duy) / ozg
                sh = math.sqrt(duy / ((1 - duy) * P) + (1 - ozg) / (ozg * N))
                lr_eksi_ci = (math.exp(math.log(lr_eksi) - 1.96 * sh),
                              math.exp(math.log(lr_eksi) + 1.96 * sh))

            # kullanicinin verdigi toplum prevalansina gore PPD/NPD (Bayes)
            ppd_pr = npd_pr = None
            if prevalans is not None and duy > 0 and ozg > 0:
                pay = duy * prevalans
                payda = pay + (1.0 - ozg) * (1.0 - prevalans)
                ppd_pr = pay / payda if payda > 0 else None
                pay2 = ozg * (1.0 - prevalans)
                payda2 = pay2 + (1.0 - duy) * prevalans
                npd_pr = pay2 / payda2 if payda2 > 0 else None

            duy_ci = _wilson(GP, P)
            ozg_ci = _wilson(GN, N)
            dog_ci = _wilson(GP + GN, P + N)
            ppd_ci = _wilson(GP, GP + YP) if (GP + YP) > 0 else (None, None)
            npd_ci = _wilson(GN, GN + YN) if (GN + YN) > 0 else (None, None)

            # ---- cizim icin egri noktalari (seyreltilir) ----
            secim = np.arange(fpr.size)
            if fpr.size > MAX_ROC_POINTS:
                secim = np.unique(np.r_[
                    np.linspace(0, fpr.size - 1, MAX_ROC_POINTS - 2).astype(int),
                    0, i_opt, fpr.size - 1])
            noktalar = [{"fpr": _f(fpr[i]), "tpr": _f(tpr[i])} for i in secim]
            opt_sira = int(np.searchsorted(secim, i_opt))

            sonuclar.append({
                "name": ad,
                "label": etiketler[ad].get("label") or ad,
                "direction": "lower" if ters else "higher",
                "n": int(P + N), "nPos": int(P), "nNeg": int(N),
                "auc": _f(auc), "se": _f(se), "seNull": _f(se0),
                "ciLow": _f(max(0.0, auc - 1.959963984540054 * se)),
                "ciHigh": _f(min(1.0, auc + 1.959963984540054 * se)),
                "z": _f(z_auc), "p": p_auc,
                "cutoff": _f(kesme),
                "sens": _f(duy), "sensLow": _fo(duy_ci[0]), "sensHigh": _fo(duy_ci[1]),
                "spec": _f(ozg), "specLow": _fo(ozg_ci[0]), "specHigh": _fo(ozg_ci[1]),
                "ppv": _fo(ppd), "ppvLow": _fo(ppd_ci[0]), "ppvHigh": _fo(ppd_ci[1]),
                "npv": _fo(npd), "npvLow": _fo(npd_ci[0]), "npvHigh": _fo(npd_ci[1]),
                "lrPos": _fo(lr_arti), "lrPosLow": _fo(lr_arti_ci[0]), "lrPosHigh": _fo(lr_arti_ci[1]),
                "lrNeg": _fo(lr_eksi), "lrNegLow": _fo(lr_eksi_ci[0]), "lrNegHigh": _fo(lr_eksi_ci[1]),
                "accuracy": _f(dogruluk), "accLow": _fo(dog_ci[0]), "accHigh": _fo(dog_ci[1]),
                "youden": _f(duy + ozg - 1.0),
                "tp": int(GP), "fn": int(YN), "fp": int(YP), "tn": int(GN),
                "ppvPrev": _fo(ppd_pr), "npvPrev": _fo(npd_pr),
                "curve": noktalar, "optIndex": opt_sira,
            })

        if not sonuclar:
            return {"error": "Hiçbir belirteç için ROC eğrisi hesaplanamadı. "
                             "Her iki grupta da en az 3 geçerli gözlem ve testte "
                             "en az iki farklı değer bulunmalıdır."}

        # ---- DeLong ile ikili AUC karsilastirmalari ----
        karsilastirmalar = []
        gecerli = [s["name"] for s in sonuclar]
        for a, bnm in itertools.combinations(gecerli, 2):
            sa, ya, oka = skorlar[a]
            sb, yb, okb = skorlar[bnm]
            ortak = oka & okb                       # iki testin de olculdugu olgular
            if int(ortak.sum()) < 8:
                continue
            xa = pd.to_numeric(df[a], errors="coerce").to_numpy(dtype=float)[ortak]
            xb = pd.to_numeric(df[bnm], errors="coerce").to_numpy(dtype=float)[ortak]
            if next(s for s in sonuclar if s["name"] == a)["direction"] == "lower":
                xa = -xa
            if next(s for s in sonuclar if s["name"] == bnm)["direction"] == "lower":
                xb = -xb
            yo = y_tum[ortak]
            if int(yo.sum()) < 3 or int((~yo).sum()) < 3:
                continue
            P2 = np.vstack([xa[yo], xb[yo]])
            N2 = np.vstack([xa[~yo], xb[~yo]])
            auc2, S2 = _delong(P2, N2)
            fark = float(auc2[0] - auc2[1])
            var = float(S2[0, 0] + S2[1, 1] - 2.0 * S2[0, 1])
            if var <= 0:
                zc, pc, lo, hi, seF = 0.0, 1.0, fark, fark, 0.0
            else:
                seF = math.sqrt(var)
                zc = fark / seF
                pc = _f(2.0 * stats.norm.sf(abs(zc)), 1.0)
                lo = fark - 1.959963984540054 * seF
                hi = fark + 1.959963984540054 * seF
            karsilastirmalar.append({
                "a": a, "b": bnm,
                "aucA": _f(auc2[0]), "aucB": _f(auc2[1]),
                "diff": _f(fark), "se": _f(seF),
                "ciLow": _f(lo), "ciHigh": _f(hi),
                "z": _f(zc), "p": _f(pc, 1.0),
                "n": int(ortak.sum()),
                "sig": bool(pc < 0.05),
            })

        return {
            "test": "ROC Eğrisi Analizi",
            "statusVar": req.statusVar,
            "positive": str(poz_deger),
            "negative": str(neg_deger),
            "prevalence": prevalans,
            "results": sonuclar,
            "comparisons": karsilastirmalar,
            "skipped": skipped,
        }
    except Exception as e:
        log.exception("roc failed")
        return {"error": f"ROC Analizi Hatası: {e}"}
    finally:
        gc.collect()


# ==========================================================================
# LOJISTIK REGRESYON
# --------------------------------------------------------------------------
# Uc model turu:
#   binary       : ikili bagimli degisken (Var/Yok)           -> Logit
#   ordinal      : sirali bagimli degisken (Dusuk/Orta/Yuksek)-> OrderedModel
#                  (orantili odds / paralel egrilik modeli)
#   multinomial  : sirasiz uc+ kategori                        -> MNLogit
#
# Kategorik bagimsiz degiskenler REFERANS KATEGORI secilerek kukla (dummy)
# kodlanir; boylece "Egitim (ref: Ilkokul)" bicimindeki SPSS duzeni uretilir.
#
# Degisken secimi: Enter (hepsi birlikte), Backward LR ve Forward LR. Adimsal
# yontemlerde bir kategorik degiskenin BUTUN kuklalari birlikte girer ya da
# birlikte cikar; tek tek islem gormeleri modeli anlamsiz hale getirirdi.
# ==========================================================================

MAX_LOJ_VARS = int(os.getenv("AIS_MAX_LOJ_VARS", "30"))
MAX_LOJ_LEVELS = int(os.getenv("AIS_MAX_LOJ_LEVELS", "12"))


class LogisticRequest(BaseModel):
    depVar: str
    modelType: Optional[str] = "binary"          # binary | ordinal | multinomial
    positive: Optional[Any] = None               # ikili modelde olay kategorisi
    reference: Optional[Any] = None              # cok kategorilide referans
    levelOrder: Optional[List[Any]] = None       # sirali modelde duzey sirasi
    numericVars: Optional[List[str]] = None
    categoricalVars: Optional[List[Dict[str, Any]]] = None   # [{"name":..,"reference":..}]
    method: Optional[str] = "enter"              # enter | backward | forward
    pEnter: Optional[float] = 0.05
    pRemove: Optional[float] = 0.10
    data: List[Dict[str, Any]]


# --------------------------------------------------------------------------
# Tasarim matrisi
# --------------------------------------------------------------------------

def _loj_tasarim(df, sayisal, kategorik):
    """Sayisal ve kukla kodlanmis sutunlardan tasarim matrisi kurar.

    Donen 'bloklar' listesi, adimsal yontemin bir degiskenin butun
    kuklalarini birlikte ele alabilmesi icindir.
    """
    parcalar, bloklar, uyarilar = [], [], []

    for v in sayisal:
        s = pd.to_numeric(df[v], errors="coerce")
        parcalar.append(pd.DataFrame({v: s}))
        bloklar.append({"ad": v, "tur": "sayisal", "sutunlar": [v], "ref": None})

    for k in kategorik:
        ad = k.get("name")
        if not ad or ad not in df.columns:
            continue
        ham = df[ad]
        duzeyler = _sorted_levels(ham)
        if len(duzeyler) < 2:
            uyarilar.append(f"{ad} tek kategoriden oluştuğu için modele alınmadı.")
            continue
        if len(duzeyler) > MAX_LOJ_LEVELS:
            uyarilar.append(f"{ad} değişkeninde {len(duzeyler)} kategori var; "
                            f"en fazla {MAX_LOJ_LEVELS} kategori işlenebilir.")
            continue
        ref = k.get("reference")
        eslesen = [g for g in duzeyler if str(g) == str(ref)]
        if ref is not None and not eslesen:
            uyarilar.append(f"{ad} için belirtilen referans kategori bulunamadı; "
                            f"ilk kategori (\u201c{duzeyler[0]}\u201d) referans alındı.")
        ref = eslesen[0] if eslesen else duzeyler[0]
        digerleri = [g for g in duzeyler if str(g) != str(ref)]
        sut = {}
        for g in digerleri:
            kolon = f"{ad}[{g}]"
            sut[kolon] = (ham.astype(str) == str(g)).astype(float)
        parcalar.append(pd.DataFrame(sut, index=df.index))
        bloklar.append({"ad": ad, "tur": "kategorik", "sutunlar": list(sut.keys()),
                        "ref": str(ref), "duzeyler": [str(g) for g in duzeyler]})

    X = pd.concat(parcalar, axis=1) if parcalar else pd.DataFrame(index=df.index)
    return X, bloklar, uyarilar


# --------------------------------------------------------------------------
# Model uydurma (uc tur icin ortak arayuz)
# --------------------------------------------------------------------------

class _BosModelSonuc(object):
    """Yalnizca sabit terim (ya da sirali modelde yalnizca esikler) iceren
    modelin sonucu. statsmodels'in OrderedModel'i sifir sutunlu tasarim
    matrisini kabul etmedigi icin bos modelin log olabilirligi analitik
    olarak verilir; uc model turunde de ayni degeri uretir."""

    def __init__(self, llf):
        self.llf = float(llf)
        self.params = np.array([])


def _bos_ll(y):
    """Yordayicisiz modelin log olabilirligi: toplam n_k * ln(n_k / n)."""
    say = pd.Series(np.asarray(y)).value_counts()
    n = float(say.sum())
    if n <= 0:
        return float("nan")
    return float(sum(c * math.log(c / n) for c in say.values if c > 0))


def _loj_fit(tur, X, y, kSinif):
    """Secilen turde modeli uydurur; yakinsamazsa None doner."""
    import warnings
    if X.shape[1] == 0:                 # yordayicisiz (bos) model
        return _BosModelSonuc(_bos_ll(y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if tur == "ordinal":
                from statsmodels.miscmodels.ordinal_model import OrderedModel
                m = OrderedModel(y, X, distr="logit") if X.shape[1] else \
                    OrderedModel(y, np.empty((len(y), 0)), distr="logit")
                return m.fit(method="bfgs", maxiter=200, disp=False)
            if tur == "multinomial":
                Xc = sm.add_constant(X, has_constant="add") if X.shape[1] else \
                    pd.DataFrame({"const": np.ones(len(y))}, index=y.index)
                return sm.MNLogit(y, Xc).fit(method="newton", maxiter=100, disp=False)
            Xc = sm.add_constant(X, has_constant="add") if X.shape[1] else \
                pd.DataFrame({"const": np.ones(len(y))}, index=y.index)
            return sm.Logit(y, Xc).fit(method="newton", maxiter=100, disp=False)
        except Exception:
            try:                        # newton takilirsa daha dayanikli yontem
                if tur == "multinomial":
                    return sm.MNLogit(y, Xc).fit(method="bfgs", maxiter=400, disp=False)
                if tur == "binary":
                    return sm.Logit(y, Xc).fit(method="bfgs", maxiter=400, disp=False)
            except Exception:
                return None
            return None


def _loj_ll(tur, X, y, kSinif):
    r = _loj_fit(tur, X, y, kSinif)
    return (None, None) if r is None else (float(r.llf), int(_loj_npar(tur, r, X, kSinif)))


def _loj_npar(tur, res, X, kSinif):
    """Modeldeki egim parametresi sayisi (sabit/esikler haric)."""
    p = X.shape[1]
    return p if tur != "multinomial" else p * (kSinif - 1)


# --------------------------------------------------------------------------
# Paralel egrilik (orantili odds) sinamasi -- Brant (1990)
# --------------------------------------------------------------------------

def _brant(X, y, kSinif, bloklar=None):
    """Brant (1990, Biometrics 46, 1171-1178) paralel egrilik sinamasi.

    Sirali bagimli degiskenin K-1 kumulatif bolunmesi icin (Y > j) ayri ayri
    ikili lojistik modeller uydurulur; orantili odds varsayimi, bu modellerin
    egim vektorlerinin birbirine esit olmasini gerektirir. Esitlik, modeller
    arasi kovaryansi Brant'in verdigi formulle hesaplanan Wald tipi bir
    ki-kare ile sinanir. Anlamli (kucuk p) sonuc varsayimin ihlal edildigini
    gosterir. Genel sinamanin yani sira her degisken icin ayri sinama da
    dondurulur.
    """
    import warnings
    n, p = X.shape
    J = int(kSinif) - 1
    if J < 2 or p < 1 or n <= p + 2:
        return None
    Xd = np.column_stack([np.ones(n), np.asarray(X, dtype=float)])   # n x (p+1)
    yv = np.asarray(y, dtype=int)
    betalar, piler = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in range(J):
            hedef = (yv > j).astype(int)
            if hedef.sum() < 2 or hedef.sum() > n - 2:
                return None
            r = None
            for yont in ("newton", "bfgs"):
                try:
                    r = sm.Logit(hedef, Xd).fit(method=yont, maxiter=500, disp=False)
                    break
                except Exception:
                    r = None
            if r is None:
                return None
            b = np.asarray(r.params, dtype=float)
            pi = np.asarray(r.predict(Xd), dtype=float)
            if not np.all(np.isfinite(b)) or not np.all(np.isfinite(pi)):
                return None
            betalar.append(b)
            piler.append(pi)

    k = p + 1
    try:
        ters = [np.linalg.inv(Xd.T @ (Xd * (piler[j] * (1.0 - piler[j]))[:, None]))
                for j in range(J)]
    except np.linalg.LinAlgError:
        return None

    V = np.zeros((J * k, J * k))
    for j in range(J):
        V[j * k:(j + 1) * k, j * k:(j + 1) * k] = ters[j]
    for j in range(J):
        for l in range(j + 1, J):
            # Cov(1{Y>j}, 1{Y>l}) = pi_l - pi_j * pi_l   (j < l oldugundan)
            w = piler[l] - piler[j] * piler[l]
            M = ters[j] @ (Xd.T @ (Xd * w[:, None])) @ ters[l]
            V[j * k:(j + 1) * k, l * k:(l + 1) * k] = M
            V[l * k:(l + 1) * k, j * k:(j + 1) * k] = M.T
    beta = np.concatenate(betalar)

    def wald(sutunlar):
        satirlar = []
        for c in sutunlar:                       # sabit terim (0) disinda
            for j in range(J - 1):
                d = np.zeros(J * k)
                d[j * k + c] = 1.0
                d[(j + 1) * k + c] = -1.0
                satirlar.append(d)
        if not satirlar:
            return None
        D = np.vstack(satirlar)
        fark = D @ beta
        try:
            orta = np.linalg.inv(D @ V @ D.T)
        except np.linalg.LinAlgError:
            return None
        ki = float(fark @ orta @ fark)
        if not math.isfinite(ki):
            return None
        sd = int(D.shape[0])
        return {"chi2": _f(max(ki, 0.0)), "df": sd,
                "p": _f(stats.chi2.sf(max(ki, 0.0), sd), 1.0)}

    genel = wald(list(range(1, k)))
    if genel is None:
        return None
    terimler = []
    if bloklar:
        adlar = list(X.columns)
        for b in bloklar:
            idx = [adlar.index(c) + 1 for c in b["sutunlar"] if c in adlar]
            t = wald(idx)
            if t:
                t["term"] = b["ad"]
                terimler.append(t)
    genel["terms"] = terimler
    return genel


# --------------------------------------------------------------------------
# Adimsal degisken secimi (olabilirlik orani olcutu)
# --------------------------------------------------------------------------

def _loj_stepwise(tur, X, y, bloklar, yontem, p_giris, p_cikis, kSinif):
    """SPSS'in "Backward LR" / "Forward LR" mantigi.

    Her adimda bir DEGISKEN (kategorikse butun kuklalariyla) modele girer ya
    da modelden cikar; olcut, o degiskeni cikarmanin/eklemenin olabilirlik
    oraninda yarattigi ki-kare degisiminin p degeridir.
    """
    adimlar = []
    if yontem == "backward":
        secili = [b["ad"] for b in bloklar]
    else:
        secili = []

    def mat(adlar):
        kolonlar = [c for b in bloklar if b["ad"] in adlar for c in b["sutunlar"]]
        return X[kolonlar] if kolonlar else X.iloc[:, :0]

    for adim in range(1, len(bloklar) * 2 + 2):
        ll_tam, k_tam = _loj_ll(tur, mat(secili), y, kSinif)
        if ll_tam is None:
            break

        if yontem == "backward":
            if not secili:
                break
            en_iyi, en_iyi_p, en_iyi_ki = None, -1.0, None
            for b in bloklar:
                if b["ad"] not in secili:
                    continue
                kalan = [a for a in secili if a != b["ad"]]
                ll_az, k_az = _loj_ll(tur, mat(kalan), y, kSinif)
                if ll_az is None:
                    continue
                ki = 2.0 * (ll_tam - ll_az)
                sd = max(k_tam - k_az, 1)
                p = float(stats.chi2.sf(max(ki, 0.0), sd))
                if p > en_iyi_p:
                    en_iyi, en_iyi_p, en_iyi_ki, en_iyi_sd = b["ad"], p, ki, sd
            if en_iyi is None or en_iyi_p <= p_cikis:
                break
            secili = [a for a in secili if a != en_iyi]
            adimlar.append({"adim": adim, "islem": "çıkarıldı", "degisken": en_iyi,
                            "ki2": _f(en_iyi_ki), "df": int(en_iyi_sd),
                            "p": _f(en_iyi_p, 1.0),
                            "kalan": list(secili)})
        else:
            adaylar = [b for b in bloklar if b["ad"] not in secili]
            if not adaylar:
                break
            en_iyi, en_iyi_p, en_iyi_ki = None, 2.0, None
            for b in adaylar:
                yeni = secili + [b["ad"]]
                ll_c, k_c = _loj_ll(tur, mat(yeni), y, kSinif)
                if ll_c is None:
                    continue
                ki = 2.0 * (ll_c - ll_tam)
                sd = max(k_c - k_tam, 1)
                p = float(stats.chi2.sf(max(ki, 0.0), sd))
                if p < en_iyi_p:
                    en_iyi, en_iyi_p, en_iyi_ki, en_iyi_sd = b["ad"], p, ki, sd
            if en_iyi is None or en_iyi_p >= p_giris:
                break
            secili = secili + [en_iyi]
            adimlar.append({"adim": adim, "islem": "eklendi", "degisken": en_iyi,
                            "ki2": _f(en_iyi_ki), "df": int(en_iyi_sd),
                            "p": _f(en_iyi_p, 1.0),
                            "kalan": list(secili)})
    return secili, adimlar


# --------------------------------------------------------------------------
# Uyum iyiligi ve siniflandirma
# --------------------------------------------------------------------------

def _hosmer_lemeshow(y, p, grup=10):
    """Hosmer-Lemeshow C istatistigi (tahmini olasiliga gore desil dilimleri)."""
    n = y.size
    if n < 2 * grup:
        grup = max(4, n // 10)
    sira = np.argsort(p, kind="mergesort")
    dilimler = np.array_split(sira, grup)
    satirlar, ki = [], 0.0
    kullanilan = 0
    for i, d in enumerate(dilimler, start=1):
        if d.size == 0:
            continue
        o1 = float(y[d].sum())
        e1 = float(p[d].sum())
        o0, e0 = d.size - o1, d.size - e1
        if e1 <= 0 or e0 <= 0:
            continue
        ki += (o1 - e1) ** 2 / e1 + (o0 - e0) ** 2 / e0
        kullanilan += 1
        satirlar.append({"dilim": i, "n": int(d.size),
                         "gozlenen1": int(round(o1)), "beklenen1": _f(e1),
                         "gozlenen0": int(round(o0)), "beklenen0": _f(e0)})
    sd = max(kullanilan - 2, 1)
    return {"chi2": _f(ki), "df": int(sd), "p": _f(stats.chi2.sf(ki, sd), 1.0),
            "groups": satirlar}


def _sinif_tablosu(y, tahmin, etiketler):
    """Gozlenen x tahmin capraz tablosu ve dogru siniflandirma yuzdeleri."""
    k = len(etiketler)
    M = np.zeros((k, k), dtype=int)
    for g, t in zip(y, tahmin):
        M[int(g), int(t)] += 1
    satirlar = []
    for i, ad in enumerate(etiketler):
        toplam = int(M[i].sum())
        satirlar.append({"gozlenen": str(ad),
                         "tahmin": [int(x) for x in M[i]],
                         "n": toplam,
                         "dogru": _f(100.0 * M[i, i] / toplam) if toplam else 0.0})
    genel = _f(100.0 * np.trace(M) / max(M.sum(), 1))
    return {"labels": [str(e) for e in etiketler], "rows": satirlar,
            "overall": genel}


# ==========================================================================

@app.post("/logistic")
def logistic(req: LogisticRequest):
    try:
        sayisal = [v for v in dict.fromkeys(req.numericVars or []) if v != req.depVar]
        kategorik = []
        gorulen = set()
        for k in (req.categoricalVars or []):
            ad = k.get("name")
            if ad and ad != req.depVar and ad not in gorulen and ad not in sayisal:
                gorulen.add(ad)
                kategorik.append(k)
        cols = [req.depVar] + sayisal + [k["name"] for k in kategorik]
        bad = _guard(req.data, cols)
        if bad:
            return bad
        if not sayisal and not kategorik:
            return {"error": "En az bir bağımsız değişken seçmelisiniz."}
        if len(sayisal) + len(kategorik) > MAX_LOJ_VARS:
            return {"error": f"Aynı anda en fazla {MAX_LOJ_VARS} bağımsız değişken "
                             f"analiz edilebilir."}

        df = _frame(req.data, cols)
        bad = _missing(df, cols)
        if bad:
            return bad

        tur = (req.modelType or "binary").strip().lower()
        if tur not in ("binary", "ordinal", "multinomial"):
            tur = "binary"
        yontem = (req.method or "enter").strip().lower()
        if yontem not in ("enter", "backward", "forward"):
            yontem = "enter"

        # ---------------- bagimli degisken ----------------
        df = df.dropna(subset=[req.depVar])
        duzeyler = _sorted_levels(df[req.depVar])
        if tur == "binary":
            if len(duzeyler) != 2:
                return {"error": f"İkili lojistik regresyon için bağımlı değişkende tam "
                                 f"olarak 2 kategori olmalıdır. Sizde {len(duzeyler)} bulundu."}
            poz = req.positive
            eslesen = [g for g in duzeyler if str(g) == str(poz)]
            poz = eslesen[0] if eslesen else duzeyler[-1]
            neg = [g for g in duzeyler if str(g) != str(poz)][0]
            siralı = [neg, poz]
        else:
            if len(duzeyler) < 3:
                return {"error": f"Seçtiğiniz model türü için bağımlı değişkende en az 3 "
                                 f"kategori olmalıdır. Sizde {len(duzeyler)} bulundu. "
                                 f"İki kategorili değişkenlerde ikili modeli kullanınız."}
            if len(duzeyler) > 8:
                return {"error": f"Bağımlı değişkende {len(duzeyler)} kategori var; "
                                 f"en fazla 8 kategori işlenebilir."}
            if tur == "ordinal" and req.levelOrder:
                istek = [str(x) for x in req.levelOrder]
                varolan = {str(g): g for g in duzeyler}
                if set(istek) == set(varolan.keys()):
                    duzeyler = [varolan[x] for x in istek]
            siralı = list(duzeyler)
            if tur == "multinomial" and req.reference is not None:
                es = [g for g in siralı if str(g) == str(req.reference)]
                if es:
                    siralı = [es[0]] + [g for g in siralı if str(g) != str(es[0])]

        kod = {str(g): i for i, g in enumerate(siralı)}
        y_tum = df[req.depVar].astype(str).map(kod)

        # ---------------- tasarim matrisi ----------------
        X_tum, bloklar, uyarilar = _loj_tasarim(df, sayisal, kategorik)
        if X_tum.shape[1] == 0:
            return {"error": "Geçerli bir bağımsız değişken kalmadı. " + " ".join(uyarilar)}

        gecerli = y_tum.notna() & X_tum.notna().all(axis=1)
        X_tum = X_tum[gecerli]
        y = y_tum[gecerli].astype(int)
        n_once, n = int(len(df)), int(len(y))
        kSinif = len(siralı)
        if n < X_tum.shape[1] + kSinif + 1:
            return {"error": f"Geçerli veri sayısı yetersiz (n = {n}). "
                             f"{X_tum.shape[1]} yordayıcı için daha fazla gözlem gereklidir."}
        for i, g in enumerate(siralı):
            if int((y == i).sum()) < 2:
                return {"error": f"“{g}” kategorisinde yeterli gözlem yok "
                                 f"({int((y == i).sum())}). Bu kategoriyi birleştirmeyi "
                                 f"düşününüz."}

        # sabit sutun ve tam baglanti kontrolu
        sabit = [c for c in X_tum.columns if X_tum[c].nunique() < 2]
        if sabit:
            return {"error": "Şu değişken(ler) tek bir değerden oluşuyor ve modele "
                             "giremez: " + ", ".join(sabit)}
        Xm = sm.add_constant(X_tum, has_constant="add").to_numpy(dtype=float)
        if np.linalg.matrix_rank(Xm) < Xm.shape[1]:
            return {"error": "Bağımsız değişkenleriniz arasında tam doğrusal ilişki var; "
                             "model tahmin edilemedi. Bu değişkenlerden birini çıkarınız "
                             "(kategorik değişkenlerin birbirini tekrar etmediğinden "
                             "emin olunuz)."}

        # ---------------- degisken secimi ----------------
        adimlar = []
        if yontem in ("backward", "forward") and len(bloklar) > 1:
            secili_ad, adimlar = _loj_stepwise(
                tur, X_tum, y, bloklar, yontem,
                float(req.pEnter or 0.05), float(req.pRemove or 0.10), kSinif)
        else:
            secili_ad = [b["ad"] for b in bloklar]
        if not secili_ad:
            return {"error": "Seçim yöntemi hiçbir değişkeni modelde tutmadı; "
                             "hiçbir yordayıcı ölçütü karşılamıyor. Enter yöntemiyle "
                             "deneyebilirsiniz."}
        kullanilan_bloklar = [b for b in bloklar if b["ad"] in secili_ad]
        kolonlar = [c for b in kullanilan_bloklar for c in b["sutunlar"]]
        X = X_tum[kolonlar]

        # ---------------- model ----------------
        res = _loj_fit(tur, X, y, kSinif)
        if res is None:
            return {"error": "Model yakınsamadı. En sık nedeni, bir yordayıcının "
                             "sonucu neredeyse kusursuz ayırmasıdır (tam ayrışma) ya da "
                             "bazı kategorilerde gözlem sayısının çok az olmasıdır."}
        bos = _loj_fit(tur, X.iloc[:, :0], y, kSinif)
        ll_tam = float(res.llf)
        ll_bos = float(bos.llf) if bos is not None else float("nan")

        kPar = _loj_npar(tur, res, X, kSinif)
        omnibus_ki = 2.0 * (ll_tam - ll_bos) if math.isfinite(ll_bos) else 0.0
        omnibus_p = _f(stats.chi2.sf(max(omnibus_ki, 0.0), max(kPar, 1)), 1.0)
        cox = 1.0 - math.exp(-omnibus_ki / n) if n > 0 else 0.0
        cox_max = 1.0 - math.exp(2.0 * ll_bos / n) if math.isfinite(ll_bos) and n > 0 else 1.0
        nagel = cox / cox_max if cox_max > 0 else 0.0

        # ---------------- katsayilar ----------------
        def _katsayi_satirlari(isimler, b, se):
            cikti = []
            for ad, bb, ss in zip(isimler, b, se):
                wald = (bb / ss) ** 2 if ss and math.isfinite(ss) and ss > 0 else 0.0
                p = float(stats.chi2.sf(wald, 1))
                orr = math.exp(bb) if -700 < bb < 700 else None
                lo = math.exp(bb - 1.959963984540054 * ss) if orr is not None and ss < 300 else None
                hi = math.exp(bb + 1.959963984540054 * ss) if orr is not None and ss < 300 else None
                cikti.append({"term": ad, "B": _f(bb), "SE": _f(ss),
                              "wald": _f(wald), "df": 1, "p": _f(p, 1.0),
                              "OR": _fo(orr), "orLow": _fo(lo), "orHigh": _fo(hi)})
            return cikti

        katsayilar, esikler, karsilastirmalar = [], [], []
        if tur == "binary":
            isim = list(sm.add_constant(X, has_constant="add").columns)
            katsayilar = _katsayi_satirlari(isim, np.asarray(res.params, float),
                                            np.asarray(res.bse, float))
        elif tur == "ordinal":
            p_all = np.asarray(res.params, float)
            se_all = np.asarray(res.bse, float)
            nx = X.shape[1]
            katsayilar = _katsayi_satirlari(list(X.columns), p_all[:nx], se_all[:nx])
            # esik (threshold) parametreleri kumulatif olceğe cevrilir
            ham = p_all[nx:]
            kesim = [float(ham[0])]
            for t in ham[1:]:
                kesim.append(kesim[-1] + math.exp(float(t)))
            for i, c in enumerate(kesim):
                esikler.append({"term": f"{siralı[i]} | {siralı[i+1]}", "B": _f(c),
                                "SE": _f(se_all[nx + i])})
        else:
            P = np.asarray(res.params, float)
            S = np.asarray(res.bse, float)
            isim = list(sm.add_constant(X, has_constant="add").columns)
            for j in range(P.shape[1]):
                karsilastirmalar.append({
                    "kategori": str(siralı[j + 1]),
                    "referans": str(siralı[0]),
                    "rows": _katsayi_satirlari(isim, P[:, j], S[:, j])})

        # ---------------- kategorik degiskenler icin BLOK Wald sinamasi --------
        # SPSS'te bir kategorik degiskenin satirinda, butun kuklalarini birlikte
        # sinayan genel bir Wald degeri yer alir. Tek tek kuklalarin p degerleri
        # "bu kategori referanstan farkli mi", blok Wald ise "bu DEGISKENIN
        # butununun etkisi var mi" sorusunu yanitlar.
        blok_testleri = {}
        if tur in ("binary", "ordinal"):
            try:
                isimler = (list(sm.add_constant(X, has_constant="add").columns)
                           if tur == "binary" else list(X.columns))
                kov = np.asarray(res.cov_params(), dtype=float)
                par = np.asarray(res.params, dtype=float)
                for b in kullanilan_bloklar:
                    # Tek kuklali (iki kategorili) degiskende de blok satiri
                    # yazilir; orada Wald degeri kuklanin kendisiyle ayni cikar
                    # ama satirin bos kalmasi tabloyu bozuk gosteriyordu.
                    if b["tur"] != "kategorik" or len(b["sutunlar"]) < 1:
                        continue
                    idx = [isimler.index(c) for c in b["sutunlar"] if c in isimler]
                    if len(idx) < 1:
                        continue
                    bb = par[idx]
                    VV = kov[np.ix_(idx, idx)]
                    try:
                        W = float(bb @ np.linalg.solve(VV, bb))
                    except np.linalg.LinAlgError:
                        continue
                    sd_b = len(idx)
                    blok_testleri[b["ad"]] = {
                        "wald": _f(W), "df": int(sd_b),
                        "p": _f(stats.chi2.sf(max(W, 0.0), sd_b), 1.0)}
            except Exception:
                blok_testleri = {}

        # ---------------- uyum, siniflandirma, ROC ----------------
        hl, roc, sinif = None, None, None
        y_np = y.to_numpy()
        if tur == "binary":
            p_hat = np.asarray(res.predict(), dtype=float)
            hl = _hosmer_lemeshow(y_np.astype(float), p_hat)
            tahmin = (p_hat >= 0.5).astype(int)
            sinif = _sinif_tablosu(y_np, tahmin, [str(siralı[0]), str(siralı[1])])
            GP = int(((y_np == 1) & (tahmin == 1)).sum())
            YN = int(((y_np == 1) & (tahmin == 0)).sum())
            YP = int(((y_np == 0) & (tahmin == 1)).sum())
            GN = int(((y_np == 0) & (tahmin == 0)).sum())
            sinif.update({"tp": GP, "fn": YN, "fp": YP, "tn": GN,
                          "sens": _f(GP / (GP + YN)) if (GP + YN) else 0.0,
                          "spec": _f(GN / (GN + YP)) if (GN + YP) else 0.0})
            # ROC: modelin tahmin ettigi olasiliklar uzerinden
            m1 = y_np == 1
            if m1.sum() >= 3 and (~m1).sum() >= 3 and np.unique(p_hat).size > 1:
                fpr, tpr, esik_r, tp_r, fp_r, P_, N_ = _roc_egrisi(p_hat, y_np.astype(float))
                auc_v, S2 = _delong(p_hat[m1].reshape(1, -1), p_hat[~m1].reshape(1, -1))
                auc = float(auc_v[0]); se_auc = float(math.sqrt(max(S2[0, 0], 0.0)))
                _, tie_sum = _ranks_with_ties(p_hat)
                NN = float(p_hat.size)
                var0 = (m1.sum() * (~m1).sum() / 12.0) * ((NN + 1.0) - tie_sum / (NN * (NN - 1.0)))
                se0 = math.sqrt(max(var0, 0.0)) / (m1.sum() * (~m1).sum()) if var0 > 0 else 0.0
                z_auc = (auc - 0.5) / se0 if se0 > 0 else 0.0
                secim = np.arange(fpr.size)
                if fpr.size > MAX_ROC_POINTS:
                    secim = np.unique(np.r_[np.linspace(0, fpr.size - 1,
                                                        MAX_ROC_POINTS - 2).astype(int),
                                            0, fpr.size - 1])
                roc = {"auc": _f(auc), "se": _f(se_auc),
                       "ciLow": _f(max(0.0, auc - 1.959963984540054 * se_auc)),
                       "ciHigh": _f(min(1.0, auc + 1.959963984540054 * se_auc)),
                       "z": _f(z_auc), "p": _f(2.0 * stats.norm.sf(abs(z_auc)), 1.0),
                       "curve": [{"fpr": _f(fpr[i]), "tpr": _f(tpr[i])} for i in secim]}
        else:
            olasilik = np.asarray(res.predict(), dtype=float)
            if olasilik.ndim == 1:
                olasilik = np.column_stack([1 - olasilik, olasilik])
            tahmin = olasilik.argmax(axis=1)
            sinif = _sinif_tablosu(y_np, tahmin, [str(g) for g in siralı])

        # ---------------- sirali modelde paralel egrilik sinamasi ----------------
        paralel = None
        if tur == "ordinal":
            paralel = _brant(X, y, kSinif, kullanilan_bloklar)

        # ---------------- tam ayrisma (separation) tespiti ----------------
        # Kusursuz ayrimda en cok olabilirlik kestirimi sonsuza gider: katsayilar
        # ve standart hatalar patlar, odds orani anlamsiz buyuklukte cikar.
        # Sayilari gizlemiyoruz ama isaretliyoruz ki yorumlanmasin.
        ayrisma = []
        for c in katsayilar:
            if abs(c["B"]) > 12 or c["SE"] > 20:
                ayrisma.append(c["term"])
        for k in karsilastirmalar:
            for c in k["rows"]:
                if abs(c["B"]) > 12 or c["SE"] > 20:
                    ayrisma.append(f"{k['kategori']}: {c['term']}")
        ayrisma = [a for a in dict.fromkeys(ayrisma) if a != "const"]
        if ayrisma:
            uyarilar.append(
                "Şu terimlerde katsayı ve standart hata olağandışı büyük: "
                + ", ".join(ayrisma)
                + ". Bu, yordayıcının sonucu neredeyse kusursuz ayırdığı (tam ya da "
                  "yarı ayrışma) durumun işaretidir; odds oranları ve güven "
                  "aralıkları yorumlanamaz. İlgili değişkeni modelden çıkarmayı ya da "
                  "kategorileri birleştirmeyi düşününüz.")

        return {
            "separation": ayrisma,
            "test": {"binary": "İkili Lojistik Regresyon",
                     "ordinal": "Sıralı (Orantılı Odds) Lojistik Regresyon",
                     "multinomial": "Çok Kategorili Lojistik Regresyon"}[tur],
            "modelType": tur, "method": yontem,
            "depVar": req.depVar,
            "levels": [str(g) for g in siralı],
            "positive": str(siralı[1]) if tur == "binary" else None,
            "negative": str(siralı[0]) if tur == "binary" else None,
            "reference": str(siralı[0]) if tur == "multinomial" else None,
            "n": n, "nDropped": int(n_once - n),
            "counts": {str(g): int((y == i).sum()) for i, g in enumerate(siralı)},
            "blocks": [{"ad": b["ad"], "tur": b["tur"], "ref": b["ref"],
                        "sutunlar": b["sutunlar"]} for b in kullanilan_bloklar],
            "excluded": [b["ad"] for b in bloklar if b["ad"] not in secili_ad],
            "steps": adimlar,
            "ll": _f(ll_tam), "llNull": _f(ll_bos),
            "minus2LL": _f(-2.0 * ll_tam),
            "omnibus": {"chi2": _f(omnibus_ki), "df": int(kPar), "p": omnibus_p},
            "coxSnell": _f(cox), "nagelkerke": _f(min(max(nagel, 0.0), 1.0)),
            "aic": _f(res.aic) if hasattr(res, "aic") else None,
            "bic": _f(res.bic) if hasattr(res, "bic") else None,
            "coefficients": katsayilar,
            "blockTests": blok_testleri,
            "thresholds": esikler,
            "comparisons": karsilastirmalar,
            "hosmerLemeshow": hl,
            "classification": sinif,
            "roc": roc,
            "parallelLines": paralel,
            "warnings": uyarilar,
        }
    except Exception as e:
        log.exception("logistic failed")
        return {"error": f"Lojistik Regresyon Hatası: {e}"}
    finally:
        gc.collect()


# ==========================================================================
# NORMALLIK SINAMASI
# ==========================================================================

@app.post("/normality")
def normality(req: NormalityRequest):
    try:
        bad = _guard(req.data, req.depVars)
        if bad:
            return bad
        df = _frame(req.data, list(req.depVars))

        results, skipped = [], []
        for var in req.depVars:
            if var not in df.columns:
                skipped.append(var)
                continue
            data = pd.to_numeric(df[var], errors="coerce").dropna().to_numpy(dtype=float)
            n = data.size
            if n < 3:
                skipped.append(var)
                continue

            mean = float(data.mean())
            median = float(np.median(data))
            vals, counts = np.unique(data, return_counts=True)
            mode = float(vals[int(np.argmax(counts))])
            std = float(data.std(ddof=1)) if n > 1 else 0.0

            base = {"varName": var, "n": int(n), "mean": mean, "median": median,
                    "mode": mode, "std": std,
                    "min": float(data.min()), "max": float(data.max()),
                    "se_skew": _f(math.sqrt(6.0 * n * (n - 1) /
                                            ((n - 2) * (n + 1) * (n + 3)))) if n > 3 else None}
            if std == 0:
                base.update({"skewness": 0.0, "kurtosis": 0.0,
                             "ks_stat": 0.0, "ks_p": 1.0,
                             "ks_method": "sabit değişken", "shapiro_W": None,
                             "shapiro_p": None, "normal": True})
                results.append(base)
                continue

            skew = _f(stats.skew(data, bias=False))
            kurt = _f(stats.kurtosis(data, bias=False))

            # Kolmogorov-Smirnov. Ortalama ve standart sapma ORNEKTEN
            # kestirildigi icin klasik K-S dagilimi gecerli degildir ve
            # p degerini oldugundan buyuk gosterir (normallik lehine yanli).
            # SPSS'in "Lilliefors anlamlilik duzeltmesi" ile raporladigi
            # duzeltme burada da uygulanir; D istatistigi ayni kalir.
            z = (data - mean) / std
            ks_stat = _f(stats.kstest(z, stats.norm.cdf).statistic)
            ks_p, ks_method, ks_bounded = None, "Lilliefors", False
            try:
                from statsmodels.stats.diagnostic import lilliefors
                d_l, p_l = lilliefors(data, dist="norm", pvalmethod="table")
                ks_stat = _f(d_l, ks_stat)
                ks_p = _f(p_l, 1.0)
                # tablo yontemi p'yi [0,001 – 0,99] araligina kirpar;
                # sinirdaysa sayfa "p < ,001" / "p > ,20" yazabilsin diye isaretle
                ks_bounded = bool(ks_p <= 0.001 or ks_p >= 0.99)
            except Exception:
                ks_p = _f(stats.kstest(z, stats.norm.cdf).pvalue, 1.0)
                ks_method = "Kolmogorov-Smirnov (düzeltmesiz, n < 4)"

            sw_W = sw_p = None
            if 3 <= n <= 5000:
                try:
                    sw = stats.shapiro(data)
                    sw_W, sw_p = _f(sw.statistic), _f(sw.pvalue, 1.0)
                except Exception:
                    pass

            # APA'da yaygin olcut: |carpiklik| < 2 ve |basiklik| < 7
            lead_p = sw_p if (sw_p is not None and n <= 50) else ks_p
            base.update({
                "skewness": skew, "kurtosis": kurt,
                "ks_stat": ks_stat, "ks_p": ks_p,
                "ks_method": ks_method,
                "ks_pBounded": ks_bounded,
                "ks_df": int(n),
                "shapiro_W": sw_W, "shapiro_p": sw_p,
                "normal": bool(lead_p is not None and lead_p >= 0.05),
            })
            results.append(base)

        return {"results": results, "skipped": skipped}
    except Exception as e:
        log.exception("normality failed")
        return {"error": f"Normallik Sınaması Hatası: {e}"}
    finally:
        gc.collect()
