"""Veri hazirlama, panel yapisi tespiti ve tanimlayici istatistikler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .utils import f

MAX_ROWS = 200_000
MAX_UNITS = 500


# --------------------------------------------------------------------------
# Panel cercevesi
# --------------------------------------------------------------------------

class Panel:
    """Dengeli/dengesiz panel veriyi tutan yardimci sinif."""

    def __init__(self, df: pd.DataFrame, id_var: str, time_var: str,
                 dep: str, indep: list[str]):
        self.id_var, self.time_var = id_var, time_var
        self.dep, self.indep = dep, list(indep)
        self.vars = [dep] + list(indep)

        # sutun adlari tekrar edebilir (orn. zaman degiskeni ayni zamanda
        # trend regresoru olarak secilmis olabilir) -> yinelenenleri temizle
        use, seen = [], set()
        for c in [id_var, time_var] + self.vars:
            if c not in seen:
                seen.add(c)
                use.append(c)
        d = df.loc[:, use].copy()
        num_cols = {c for c in self.vars + [time_var] if c != id_var}
        for c in num_cols:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d[id_var] = d[id_var].astype(str).str.strip()

        n_before = len(d)
        d = d.dropna()
        self.dropped_na = n_before - len(d)

        d = d.sort_values([id_var, time_var], kind="mergesort").reset_index(drop=True)
        # ayni birim-zaman ikilisinden birden fazla varsa ilkini al
        dup = d.duplicated([id_var, time_var])
        self.dropped_dup = int(dup.sum())
        d = d.loc[~dup].reset_index(drop=True)

        self.df = d
        self.units = list(pd.unique(d[id_var]))
        self.times = sorted(pd.unique(d[time_var]).tolist())
        self.N = len(self.units)
        self.T = len(self.times)
        self.nobs = len(d)
        counts = d.groupby(id_var, observed=True).size()
        self.t_per_unit = {str(k): int(v) for k, v in counts.items()}
        self.balanced = bool(counts.nunique() == 1 and counts.iloc[0] == self.T)
        self.t_min = int(counts.min()) if len(counts) else 0
        self.t_max = int(counts.max()) if len(counts) else 0

    # -- erisimciler -------------------------------------------------------
    def y(self):
        return self.df[self.dep].to_numpy(float)

    def X(self):
        return self.df[self.indep].to_numpy(float)

    def ids(self):
        return self.df[self.id_var].to_numpy()

    def times_arr(self):
        return self.df[self.time_var].to_numpy(float)

    def series(self, var):
        """Her birim icin (birim_adi, seri) listesi -- zaman sirasinda."""
        out = []
        for u, g in self.df.groupby(self.id_var, sort=False, observed=True):
            out.append((str(u), g[var].to_numpy(float)))
        return out

    def wide(self, var):
        """(T x N) matris; dengesizlikte NaN."""
        p = self.df.pivot_table(index=self.time_var, columns=self.id_var,
                                values=var, aggfunc="first")
        return p.sort_index()

    def balanced_subset(self):
        """Sadece tam gozlemli zaman noktalarindan olusan dengeli alt panel."""
        piv = self.df.pivot_table(index=self.time_var, columns=self.id_var,
                                  values=self.dep, aggfunc="first").sort_index()
        keep_t = piv.dropna(axis=0, how="any").index
        if len(keep_t) < 4:
            piv2 = piv.dropna(axis=1, how="any")
            if piv2.shape[1] >= 2:
                sub = self.df[self.df[self.id_var].isin(piv2.columns)]
                return Panel(sub, self.id_var, self.time_var, self.dep, self.indep)
        sub = self.df[self.df[self.time_var].isin(keep_t)]
        cnt = sub.groupby(self.id_var, observed=True).size()
        full = cnt[cnt == len(keep_t)].index
        sub = sub[sub[self.id_var].isin(full)]
        if len(sub) < 8:
            return self
        return Panel(sub, self.id_var, self.time_var, self.dep, self.indep)

    def info(self):
        return {
            "idVar": self.id_var, "timeVar": self.time_var,
            "depVar": self.dep, "indepVars": self.indep,
            "N": self.N, "T": self.T, "nobs": self.nobs,
            "balanced": self.balanced,
            "tMin": self.t_min, "tMax": self.t_max,
            "timeStart": self.times[0] if self.times else None,
            "timeEnd": self.times[-1] if self.times else None,
            "droppedMissing": self.dropped_na,
            "droppedDuplicate": self.dropped_dup,
            "unitsPreview": [str(u) for u in self.units[:25]],
            "tPerUnitMin": self.t_min, "tPerUnitMax": self.t_max,
        }


# --------------------------------------------------------------------------
# Otomatik panel yapisi tespiti (sunucu tarafi dogrulama)
# --------------------------------------------------------------------------

ID_HINTS = ["id", "birim", "ulke", "ülke", "country", "firma", "firm", "company",
            "sirket", "şirket", "il", "ilce", "ilçe", "city", "bolge", "bölge",
            "region", "state", "entity", "unit", "panel", "kod", "code", "banka",
            "bank", "sektor", "sektör", "sector", "hasta", "ogrenci", "öğrenci",
            "katilimci", "katılımcı", "subject", "case", "no", "kimlik"]
TIME_HINTS = ["year", "yil", "yıl", "time", "zaman", "donem", "dönem", "period",
              "date", "tarih", "quarter", "ceyrek", "çeyrek", "month", "ay",
              "wave", "dalga", "t", "olcum", "ölçüm", "hafta", "week"]


def detect_structure(df: pd.DataFrame):
    """Sutun isimleri ve veri desenlerinden id / zaman sutununu tahmin eder."""
    cols = list(df.columns)
    n = len(df)
    scores_id, scores_time = {}, {}

    for c in cols:
        s = df[c]
        name = str(c).strip().lower()
        nun = s.nunique(dropna=True)
        if nun < 2 or nun == n:
            id_s, tm_s = 0.0, 0.0
        else:
            id_s = tm_s = 0.0
        numeric = pd.to_numeric(s, errors="coerce")
        num_ratio = float(numeric.notna().mean())

        # --- id skoru
        if 2 <= nun <= min(MAX_UNITS, max(2, n // 2)):
            id_s += 2.0
            if n % nun == 0:
                id_s += 1.0
            if not (num_ratio > 0.9 and numeric.dropna().is_monotonic_increasing):
                id_s += 0.5
        if any(h == name for h in ID_HINTS):
            id_s += 4.0
        elif any(h in name for h in ID_HINTS):
            id_s += 2.0
        if num_ratio < 0.5:
            id_s += 1.5  # metinsel

        # --- zaman skoru
        if num_ratio > 0.9:
            v = numeric.dropna()
            tm_s += 1.0
            if 2 <= nun <= 400:
                tm_s += 1.5
            # yil benzeri araliklar
            if v.between(1800, 2200).mean() > 0.9:
                tm_s += 3.0
            # birim icinde artan mi?
            uniq = sorted(v.unique())
            if len(uniq) > 1:
                d = np.diff(uniq)
                if np.allclose(d, d[0]):
                    tm_s += 1.5
        if any(h == name for h in TIME_HINTS):
            tm_s += 4.0
        elif any(h in name for h in TIME_HINTS):
            tm_s += 2.0

        scores_id[c] = id_s
        scores_time[c] = tm_s

    id_var = max(scores_id, key=scores_id.get) if scores_id else None
    time_pool = {k: v for k, v in scores_time.items() if k != id_var}
    time_var = max(time_pool, key=time_pool.get) if time_pool else None

    ok = False
    if id_var and time_var:
        try:
            dd = df[[id_var, time_var]].dropna()
            ok = bool(not dd.duplicated().any() and dd[id_var].nunique() >= 2)
        except Exception:
            ok = False
    return {"idVar": id_var, "timeVar": time_var, "confident": ok,
            "idScore": f(scores_id.get(id_var, 0), 2),
            "timeScore": f(scores_time.get(time_var, 0), 2)}


# --------------------------------------------------------------------------
# Tanimlayici istatistikler
# --------------------------------------------------------------------------

def descriptives(pnl: Panel):
    rows = []
    for v in pnl.vars:
        x = pnl.df[v].to_numpy(float)
        x = x[np.isfinite(x)]
        if x.size < 3:
            continue
        jb, jbp = stats.jarque_bera(x)[:2]
        # within / between ayrisimi
        g = pnl.df.groupby(pnl.id_var, observed=True)[v]
        means = g.mean()
        between_sd = float(means.std(ddof=1)) if len(means) > 1 else np.nan
        dm = pnl.df[v].to_numpy(float) - pnl.df[pnl.id_var].map(means).to_numpy(float)
        within_sd = float(np.nanstd(dm, ddof=1))
        rows.append({
            "variable": v,
            "n": int(x.size),
            "mean": f(np.mean(x)), "median": f(np.median(x)),
            "sd": f(np.std(x, ddof=1)),
            "min": f(np.min(x)), "max": f(np.max(x)),
            "skewness": f(stats.skew(x)), "kurtosis": f(stats.kurtosis(x, fisher=False)),
            "jb": f(jb), "jbP": f(jbp),
            "betweenSd": f(between_sd), "withinSd": f(within_sd),
            "cv": f(np.std(x, ddof=1) / np.mean(x)) if np.mean(x) != 0 else None,
        })
    return rows


def correlation(pnl: Panel):
    sub = pnl.df[pnl.vars]
    c = sub.corr(method="pearson")
    n = len(sub)
    out = {"vars": pnl.vars, "matrix": [], "pmatrix": []}
    for a in pnl.vars:
        out["matrix"].append([f(c.loc[a, b], 4) for b in pnl.vars])
        prow = []
        for b in pnl.vars:
            if a == b:
                prow.append(0.0)
            else:
                r = c.loc[a, b]
                if not np.isfinite(r) or abs(r) >= 1 or n <= 2:
                    prow.append(None)
                else:
                    t = r * np.sqrt((n - 2) / (1 - r ** 2))
                    prow.append(f(2 * stats.t.sf(abs(t), n - 2), 4))
        out["pmatrix"].append(prow)
    return out
