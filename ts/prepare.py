"""Zaman sutunu tespiti, frekans cikarimi ve tanimlayici istatistikler."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

from panel.utils import f

MAX_ROWS = 100_000
MIN_OBS = 12

TIME_HINTS = ["tarih", "date", "zaman", "time", "yil", "yıl", "year", "donem",
              "dönem", "period", "ay", "month", "ceyrek", "çeyrek", "quarter",
              "hafta", "week", "gun", "gün", "day", "t", "index", "obs"]

FREQ_LABEL = {"A": "Yıllık", "Q": "Çeyreklik", "M": "Aylık",
              "W": "Haftalık", "D": "Günlük", "?": "Belirsiz"}
FREQ_PERIOD = {"A": 1, "Q": 4, "M": 12, "W": 52, "D": 7, "?": 1}


# --------------------------------------------------------------------------
# Zaman sutunu tespiti
# --------------------------------------------------------------------------

def detect_time_column(df: pd.DataFrame):
    """Sutun isimleri ve veri desenlerinden zaman sutununu tahmin eder."""
    n = len(df)
    scores = {}
    for c in df.columns:
        s = df[c]
        name = str(c).strip().lower()
        sc = 0.0
        nun = s.nunique(dropna=True)

        if nun == n and n > 3:
            sc += 2.0                       # her satir farkli -> zaman adayi
        num = pd.to_numeric(s, errors="coerce")
        num_ratio = float(num.notna().mean())

        if num_ratio > 0.9:
            v = num.dropna()
            if v.is_monotonic_increasing:
                sc += 2.5
            if v.between(1800, 2200).mean() > 0.9:
                sc += 3.0                   # yil gibi
            if v.between(20000, 60000).mean() > 0.9:
                sc += 1.5                   # Excel seri tarihi
            u = np.sort(v.unique())
            if u.size > 2:
                d = np.diff(u)
                if np.allclose(d, d[0]):
                    sc += 1.5
        else:
            txt = s.dropna().astype(str)
            if txt.size:
                pat = txt.str.match(
                    r"^\s*(\d{4})\s*[-_/.\s]?\s*(Q[1-4]|[01]?\d|\d{1,2})?\s*$",
                    case=False).mean()
                if pat > 0.8:
                    sc += 2.5
                try:
                    parsed = pd.to_datetime(txt, errors="coerce", dayfirst=True)
                    if parsed.notna().mean() > 0.9:
                        sc += 3.0
                except Exception:
                    pass

        if name in TIME_HINTS:
            sc += 4.0
        elif any(h in name for h in TIME_HINTS if len(h) > 2):
            sc += 2.0
        scores[c] = sc

    if not scores:
        return {"timeVar": None, "confident": False, "score": 0}
    best = max(scores, key=scores.get)
    return {"timeVar": best, "confident": bool(scores[best] >= 4.0),
            "score": f(scores[best], 2),
            "scores": {str(k): f(v, 2) for k, v in scores.items()}}


# --------------------------------------------------------------------------
# Zaman ayristirma
# --------------------------------------------------------------------------

def _parse_time(values):
    """(etiketler, siralama_anahtari, frekans) dondurur."""
    raw = pd.Series(values)
    num = pd.to_numeric(raw, errors="coerce")

    # 1) Yil olarak sayisal
    if num.notna().mean() > 0.95:
        v = num.astype(float)
        if v.between(1500, 2200).mean() > 0.9:
            labels = [str(int(x)) for x in v]
            return labels, v.to_numpy(float), "A"
        if v.between(20000, 60000).mean() > 0.9:      # Excel seri tarihi
            dt = pd.to_datetime(v, unit="D", origin="1899-12-30", errors="coerce")
            return _from_datetime(dt)
        # duz sira numarasi
        return [str(x) for x in v], v.to_numpy(float), "?"

    txt = raw.astype(str).str.strip()

    # 2) YYYYQn
    q = txt.str.extract(r"^(\d{4})\s*[-_/.\s]?\s*[Qq]([1-4])$")
    if q[0].notna().mean() > 0.9:
        yr = q[0].astype(float); qt = q[1].astype(float)
        key = yr + (qt - 1) / 4.0
        return [f"{int(a)}Q{int(b)}" for a, b in zip(yr, qt)], key.to_numpy(float), "Q"

    # 3) Tarih — hem ISO (yyyy-mm-dd) hem gun-onceli (gg.aa.yyyy) denenir ve
    #    daha cok degeri cozebilen bicim secilir. dayfirst=True tek basina
    #    ISO tarihlerin bir kismini bozdugu icin iki deneme sarttir.
    best_dt, best_ok = None, 0.0
    for kwargs in ({"dayfirst": False}, {"dayfirst": True}):
        try:
            cand = pd.to_datetime(txt, errors="coerce", **kwargs)
        except Exception:
            continue
        ok = float(cand.notna().mean())
        if ok > best_ok:
            best_dt, best_ok = cand, ok
    if best_dt is not None and best_ok > 0.9:
        return _from_datetime(best_dt)

    # 4) Coz(e)medi -> sira
    return list(txt), np.arange(len(txt), dtype=float), "?"


def _days_since_epoch(dt: pd.Series) -> np.ndarray:
    """Tarihleri gun cinsinden kayan noktali sayiya cevirir.

    pandas surumune gore datetime64 birimi nanosaniye, mikrosaniye veya
    saniye olabildiginden ham int64 degerleri birim varsayimiyla bolmek
    hataya yol acar; fark alma yontemi birimden bagimsizdir.
    """
    dt = pd.to_datetime(dt)
    delta = dt - pd.Timestamp("1970-01-01")
    return (delta.dt.total_seconds() / 86400.0).to_numpy(float)


def _from_datetime(dt: pd.Series):
    dt = pd.to_datetime(dt)
    key = _days_since_epoch(dt)
    d = np.diff(np.sort(key[np.isfinite(key)]))
    med = float(np.median(d)) if d.size else 0.0   # gun
    if med >= 300:
        freq, labels = "A", [f"{x.year}" for x in dt]
    elif med >= 80:
        freq, labels = "Q", [f"{x.year}Q{((x.month - 1) // 3) + 1}" for x in dt]
    elif med >= 25:
        freq, labels = "M", [f"{x.year}-{x.month:02d}" for x in dt]
    elif med >= 6:
        freq, labels = "W", [x.strftime("%Y-%m-%d") for x in dt]
    else:
        freq, labels = "D", [x.strftime("%Y-%m-%d") for x in dt]
    return labels, key, freq


# --------------------------------------------------------------------------
# Veri kabi
# --------------------------------------------------------------------------

class TSData:
    """Analize hazir zaman serisi veri kabi."""

    def __init__(self, df: pd.DataFrame, time_var: str, variables: list[str],
                 freq_override=None):
        self.time_var = time_var
        self.vars = list(dict.fromkeys(variables))

        use = list(dict.fromkeys([time_var] + self.vars))
        d = df.loc[:, use].copy()
        for c in self.vars:
            d[c] = pd.to_numeric(d[c], errors="coerce")

        labels, key, freq = _parse_time(d[time_var])
        d["__lab"] = labels
        d["__key"] = key

        n0 = len(d)
        d = d.dropna(subset=self.vars + ["__key"])
        self.dropped_na = n0 - len(d)

        d = d.sort_values("__key", kind="mergesort")
        dup = d.duplicated("__key")
        self.dropped_dup = int(dup.sum())
        d = d.loc[~dup].reset_index(drop=True)

        self.df = d
        self.labels = list(d["__lab"])
        self.n = len(d)
        self.freq = freq_override or freq
        self.freq_label = FREQ_LABEL.get(self.freq, "Belirsiz")
        self.period = FREQ_PERIOD.get(self.freq, 1)

        # esit araliklilik
        k = d["__key"].to_numpy(float)
        gaps = np.diff(k)
        self.regular = bool(gaps.size == 0 or np.allclose(gaps, gaps[0], rtol=0.05))

    def y(self, var):
        return self.df[var].to_numpy(float)

    def matrix(self, variables=None):
        cols = variables or self.vars
        return self.df[cols].to_numpy(float)

    def info(self):
        return {
            "timeVar": self.time_var, "vars": self.vars,
            "n": self.n, "freq": self.freq, "freqLabel": self.freq_label,
            "period": self.period, "regular": self.regular,
            "start": self.labels[0] if self.labels else None,
            "end": self.labels[-1] if self.labels else None,
            "droppedMissing": self.dropped_na,
            "droppedDuplicate": self.dropped_dup,
        }

    def plot_series(self):
        """Grafikler icin ham seri verisi."""
        out = []
        for v in self.vars:
            y = self.y(v)
            dy = np.diff(y)
            out.append({
                "name": v,
                "labels": self.labels,
                "values": [f(x, 6) for x in y],
                "diffLabels": self.labels[1:],
                "diffValues": [f(x, 6) for x in dy],
            })
        return out


# --------------------------------------------------------------------------
# Tanimlayici istatistikler
# --------------------------------------------------------------------------

def descriptives(ts: TSData):
    rows = []
    for v in ts.vars:
        x = ts.y(v)
        x = x[np.isfinite(x)]
        if x.size < 3:
            continue
        jb, jbp = stats.jarque_bera(x)[:2]
        rows.append({
            "variable": v, "n": int(x.size),
            "mean": f(np.mean(x)), "median": f(np.median(x)),
            "max": f(np.max(x)), "min": f(np.min(x)),
            "sd": f(np.std(x, ddof=1)),
            "skewness": f(stats.skew(x)),
            "kurtosis": f(stats.kurtosis(x, fisher=False)),
            "jb": f(jb), "jbP": f(jbp),
            "normal": bool(jbp >= 0.05),
            "sum": f(np.sum(x)),
            "sumSqDev": f(float(((x - x.mean()) ** 2).sum())),
            "cv": f(np.std(x, ddof=1) / np.mean(x)) if np.mean(x) != 0 else None,
        })
    return rows


def correlation(ts: TSData):
    if len(ts.vars) < 2:
        return None
    c = ts.df[ts.vars].corr()
    n = ts.n
    out = {"vars": ts.vars, "matrix": [], "pmatrix": []}
    for a in ts.vars:
        out["matrix"].append([f(c.loc[a, b], 4) for b in ts.vars])
        prow = []
        for b in ts.vars:
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
