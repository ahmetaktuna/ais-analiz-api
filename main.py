"""
AIS Akademi - Istatistiksel Analiz API
======================================
Uc noktalar:
    GET  /ping
    POST /analyze          -> coklu dogrusal regresyon
    POST /ttest-multiple   -> bagimsiz orneklem t-testi (coklu degisken)
    POST /anova-multiple   -> tek yonlu ANOVA + post-hoc (coklu degisken)
    POST /normality        -> normallik sinamasi ve betimsel istatistikler

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
