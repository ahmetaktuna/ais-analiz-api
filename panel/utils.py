"""Ortak matematiksel yardimcilar."""
from __future__ import annotations

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# Temel EKK
# --------------------------------------------------------------------------


class OLSResult:
    """Hafif EKK sonuc tasiyicisi."""

    __slots__ = (
        "beta", "se", "tstat", "pval", "resid", "fitted", "s2", "cov",
        "nobs", "k", "df_resid", "r2", "r2_adj", "fstat", "fpval", "XtXi",
    )

    def __init__(self, beta, se, tstat, pval, resid, fitted, s2, cov,
                 nobs, k, df_resid, r2, r2_adj, fstat, fpval, XtXi):
        self.beta, self.se, self.tstat, self.pval = beta, se, tstat, pval
        self.resid, self.fitted, self.s2, self.cov = resid, fitted, s2, cov
        self.nobs, self.k, self.df_resid = nobs, k, df_resid
        self.r2, self.r2_adj = r2, r2_adj
        self.fstat, self.fpval = fstat, fpval
        self.XtXi = XtXi


def ols(y, X, has_const=True):
    """Sade EKK tahmini. y: (n,), X: (n,k) tasarim matrisi."""
    y = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != y.shape[0]:
        X = X.T
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    beta = XtXi @ (X.T @ y)
    fitted = X @ beta
    resid = y - fitted
    df_resid = max(n - k, 1)
    ssr = float(resid @ resid)
    s2 = ssr / df_resid
    cov = s2 * XtXi
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, beta / se, np.nan)
    pval = 2.0 * stats.t.sf(np.abs(tstat), df_resid)

    if has_const:
        sst = float(((y - y.mean()) ** 2).sum())
        kx = k - 1
    else:
        sst = float((y ** 2).sum())
        kx = k
    r2 = 1.0 - ssr / sst if sst > 0 else np.nan
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / df_resid if np.isfinite(r2) else np.nan
    if kx > 0 and np.isfinite(r2) and r2 < 1:
        fstat = (r2 / kx) / ((1 - r2) / df_resid)
        fpval = float(stats.f.sf(fstat, kx, df_resid))
    else:
        fstat, fpval = np.nan, np.nan

    return OLSResult(beta, se, tstat, pval, resid, fitted, s2, cov,
                     n, k, df_resid, r2, r2_adj, fstat, fpval, XtXi)


def add_const(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.column_stack([np.ones(X.shape[0]), X])


# --------------------------------------------------------------------------
# Zaman serisi yardimcilari
# --------------------------------------------------------------------------

def adf_regression(y, lags=1, trend="c"):
    """
    Genisletilmis Dickey-Fuller regresyonu.
      dy_t = a (+ b*t) + rho*y_{t-1} + sum_{j=1..p} g_j * dy_{t-j} + e_t
    Donen: (t_rho, rho, se_rho, resid, nobs, k)
    """
    y = np.asarray(y, dtype=float).ravel()
    T = y.size
    p = int(max(lags, 0))
    if T < p + 5:
        return np.nan, np.nan, np.nan, None, 0, 0

    dy = np.diff(y)
    m = dy.size
    if m - p < 4:
        return np.nan, np.nan, np.nan, None, 0, 0

    Y = dy[p:]
    cols = [y[p:-1]]                       # y_{t-1}
    for j in range(1, p + 1):
        cols.append(dy[p - j: m - j])      # dy_{t-j}
    Z = np.column_stack(cols)

    n = Y.size
    det = []
    if trend in ("c", "ct"):
        det.append(np.ones(n))
    if trend == "ct":
        det.append(np.arange(1, n + 1, dtype=float))
    if det:
        Z = np.column_stack([np.column_stack(det), Z])
        rho_idx = len(det)
    else:
        rho_idx = 0

    if Z.shape[0] <= Z.shape[1] + 1:
        return np.nan, np.nan, np.nan, None, 0, 0

    try:
        r = ols(Y, Z, has_const=(trend in ("c", "ct")))
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, None, 0, 0

    return (float(r.tstat[rho_idx]), float(r.beta[rho_idx]),
            float(r.se[rho_idx]), r.resid, r.nobs, r.k)


def adf_tstat(y, lags=1, trend="c"):
    return adf_regression(y, lags, trend)[0]


def mackinnon_adf_pvalue(tstat, trend="c", N=1):
    """
    MacKinnon (1994/2010) yaklasik ADF p-degeri.
    statsmodels varsa onun tablosunu kullanir, yoksa normal yaklasim doner.
    """
    if not np.isfinite(tstat):
        return np.nan
    try:
        from statsmodels.tsa.adfvalues import mackinnonp
        regression = {"n": "nc", "c": "c", "ct": "ct"}.get(trend, "c")
        return float(mackinnonp(tstat, regression=regression, N=N))
    except Exception:
        return float(stats.norm.cdf(tstat))


def phillips_perron(y, trend="c", lags=None):
    """
    Phillips-Perron Z-tau istatistigi (Newey-West duzeltmeli).
    Donen: (Ztau, pvalue)
    """
    y = np.asarray(y, dtype=float).ravel()
    T = y.size
    if T < 6:
        return np.nan, np.nan

    Y = np.diff(y)
    n = Y.size
    ylag = y[:-1]
    det = [np.ones(n)] if trend in ("c", "ct") else []
    if trend == "ct":
        det.append(np.arange(1, n + 1, dtype=float))
    Z = np.column_stack(det + [ylag]) if det else ylag.reshape(-1, 1)
    idx = len(det)

    try:
        r = ols(Y, Z, has_const=(trend in ("c", "ct")))
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    u = r.resid
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** 0.25))
    lags = int(max(0, min(lags, n - 2)))

    gamma0 = float(u @ u) / n
    lam2 = gamma0
    for j in range(1, lags + 1):
        w = 1.0 - j / (lags + 1.0)
        lam2 += 2.0 * w * float(u[j:] @ u[:-j]) / n
    if gamma0 <= 0 or lam2 <= 0:
        return np.nan, np.nan

    t_rho = float(r.tstat[idx])
    se_rho = float(r.se[idx])
    ztau = (np.sqrt(gamma0 / lam2) * t_rho
            - 0.5 * (lam2 - gamma0) * n * se_rho / (np.sqrt(lam2) * np.sqrt(gamma0)))
    return float(ztau), mackinnon_adf_pvalue(ztau, trend=trend, N=1)


def newey_west_lrv(u, lags=None):
    """Tek degiskenli uzun donem varyans (Bartlett cekirdegi)."""
    u = np.asarray(u, dtype=float).ravel()
    n = u.size
    if n < 3:
        return np.nan
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** 0.25))
    lags = int(max(0, min(lags, n - 2)))
    g0 = float(u @ u) / n
    lrv = g0
    for j in range(1, lags + 1):
        w = 1.0 - j / (lags + 1.0)
        lrv += 2.0 * w * float(u[j:] @ u[:-j]) / n
    return float(lrv)


def newey_west_lrcov(U, lags=None):
    """Cok degiskenli uzun donem kovaryans matrisi. U: (n, m)."""
    U = np.atleast_2d(np.asarray(U, dtype=float))
    if U.shape[0] < U.shape[1]:
        U = U.T
    n, m = U.shape
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** 0.25))
    lags = int(max(0, min(lags, n - 2)))
    Om = U.T @ U / n
    for j in range(1, lags + 1):
        w = 1.0 - j / (lags + 1.0)
        G = U[j:].T @ U[:-j] / n
        Om = Om + w * (G + G.T)
    return Om


def bic_lag_select(y, maxlag, trend="c"):
    """ADF regresyonu icin BIC ile gecikme secimi."""
    y = np.asarray(y, dtype=float).ravel()
    best, best_bic = 0, np.inf
    for p in range(0, int(maxlag) + 1):
        t, rho, se, resid, n, k = adf_regression(y, p, trend)
        if resid is None or n <= k + 1:
            continue
        ssr = float(resid @ resid)
        if ssr <= 0:
            continue
        bic = n * np.log(ssr / n) + k * np.log(n)
        if bic < best_bic:
            best_bic, best = bic, p
    return best


# --------------------------------------------------------------------------
# Bicimlendirme
# --------------------------------------------------------------------------

def f(x, nd=4):
    """JSON'a guvenli float donusumu."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return round(v, nd)


def star(p):
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def safe(obj):
    """numpy tiplerini saf Python'a cevirir (JSON serilestirme icin)."""
    if isinstance(obj, dict):
        return {k: safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [safe(v) for v in obj.tolist()]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
