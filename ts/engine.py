"""Zaman serisi analizi orkestrasyonu ve APA formatinda Turkce anlatim uretimi."""
from __future__ import annotations

import time
import traceback

import numpy as np
import pandas as pd

from panel.utils import safe
from . import arima as arima_mod
from . import coint as coint_mod
from . import garch as garch_mod
from . import stationarity as stat_mod
from . import varmod
from .prepare import (MAX_ROWS, MIN_OBS, TSData, correlation, descriptives,
                      detect_time_column)

ENGINE = {
    "name": "AIS Akademi Zaman Serisi Analizi Motoru",
    "version": "1.0.0",
    "python": "3.11",
    "libraries": ["NumPy", "pandas", "SciPy", "statsmodels", "arch"],
}

DEFAULT_MODULES = {
    "descriptives": True,
    "stationarity": True,
    "arima": False,
    "cointegration": False,
    "var": False,
    "garch": False,
}


class AnalysisError(Exception):
    pass


def _to_frame(data):
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, list):
        if not data:
            raise AnalysisError("Veri seti boş.")
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame(data)
    raise AnalysisError("Veri biçimi tanınmadı.")


# ==========================================================================

def analyze(payload: dict) -> dict:
    t0 = time.time()
    df = _to_frame(payload.get("data"))
    if len(df) > MAX_ROWS:
        raise AnalysisError(f"Veri seti çok büyük (>{MAX_ROWS:,} satır).")

    detected = detect_time_column(df)
    time_var = payload.get("timeVar") or detected["timeVar"]
    if not time_var:
        raise AnalysisError("Zaman değişkeni belirlenemedi.")
    if time_var not in df.columns:
        raise AnalysisError(f"'{time_var}' sütunu veri setinde bulunamadı.")

    variables = list(payload.get("vars") or [])
    if not variables:
        raise AnalysisError("En az bir seri (değişken) seçilmelidir.")
    if time_var in variables:
        raise AnalysisError("Zaman sütunu analiz değişkeni olarak seçilemez.")
    missing = [c for c in variables if c not in df.columns]
    if missing:
        raise AnalysisError("Şu sütunlar bulunamadı: " + ", ".join(missing))

    opts = payload.get("options") or {}
    ts = TSData(df, time_var, variables, freq_override=opts.get("freq"))

    if ts.n < MIN_OBS:
        raise AnalysisError(
            f"Zaman serisi analizi için en az {MIN_OBS} gözlem gereklidir "
            f"(mevcut: {ts.n}).")

    dep = payload.get("depVar") or variables[0]
    if dep not in variables:
        dep = variables[0]
    indep = [v for v in variables if v != dep]

    mods = dict(DEFAULT_MODULES)
    mods.update(payload.get("modules") or {})

    # Adim adim (sirali) calisma modunda onceki adimin ciktilari "hints" ile
    # tasinir; boylece her istek yalnizca kendi modulunu hesaplar.
    hints = payload.get("hints") or {}
    skip_plot = bool(payload.get("skipPlot"))

    res = {
        "ok": True, "engine": ENGINE, "meta": ts.info(),
        "detection": detected, "modules": mods, "options": dict(opts),
        "depVar": dep, "indepVars": indep,
        "warnings": [], "errors": {},
    }
    if not skip_plot:
        res["plotSeries"] = ts.plot_series()
    if ts.dropped_na:
        res["warnings"].append(
            f"{ts.dropped_na} gözlem eksik veri (NaN) nedeniyle analiz dışı bırakıldı.")
    if ts.dropped_dup:
        res["warnings"].append(
            f"{ts.dropped_dup} yinelenen zaman kaydı çıkarıldı.")
    if not ts.regular:
        res["warnings"].append(
            "Zaman aralıkları eşit değildir; gözlemler sıralı kabul edilerek "
            "analiz edilmiştir. Sonuçları dikkatle yorumlayınız.")
    if ts.freq == "?":
        res["warnings"].append(
            "Veri frekansı otomatik belirlenemedi; mevsimsel modelleme devre dışı "
            "bırakılmıştır. Gelişmiş seçeneklerden frekansı elle seçebilirsiniz.")

    # ---------------- Tanimlayici ----------------
    if mods.get("descriptives", True):
        try:
            res["descriptives"] = descriptives(ts)
            res["correlation"] = correlation(ts)
        except Exception as exc:
            res["errors"]["descriptives"] = str(exc)

    # ---------------- Duraganlik ----------------
    if mods.get("stationarity", True):
        try:
            res["stationarity"] = stat_mod.run_all(
                ts, trend=opts.get("urTrend", "c"),
                maxlag=opts.get("urMaxLag"),
                za_regression=opts.get("zaRegression", "ct"),
                tests=tuple(opts.get("urTests", ["adf", "pp", "kpss", "za"])))
        except Exception as exc:
            res["errors"]["stationarity"] = str(exc)
            res["errors"]["stationarityTrace"] = traceback.format_exc(limit=3)

    orders = (res.get("stationarity") or {}).get("orders", {}) \
        or (hints.get("orders") or {})

    # ---------------- ARIMA / SARIMA ----------------
    if mods.get("arima"):
        try:
            d_hint = None
            o = orders.get(dep)
            if o == "I(0)":
                d_hint = 0
            elif o == "I(1)":
                d_hint = 1
            if opts.get("arimaD") not in (None, "auto"):
                d_hint = int(opts["arimaD"])
            res["arima"] = arima_mod.run(
                ts, dep, d=d_hint,
                max_p=int(opts.get("arimaMaxP", 3)),
                max_q=int(opts.get("arimaMaxQ", 3)),
                seasonal=bool(opts.get("seasonal", True)),
                horizon=int(opts.get("forecastH", 5)),
                ic=str(opts.get("arimaIC", "aicc")),
                time_budget=float(opts.get("arimaTimeBudget", 45) or 45))
        except Exception as exc:
            res["errors"]["arima"] = str(exc)
            res["errors"]["arimaTrace"] = traceback.format_exc(limit=3)

    # ---------------- Esbutunlesme ----------------
    if mods.get("cointegration"):
        try:
            res["cointegration"] = coint_mod.run_all(
                ts, dep=dep, indep=indep,
                det_order=int(opts.get("johansenDet", 0)),
                k_ar_diff=int(opts.get("johansenLags", 1)),
                case=int(opts.get("ardlCase", 3)),
                max_p=int(opts.get("ardlMaxP", 4)),
                max_q=int(opts.get("ardlMaxQ", 4)),
                do_ardl=bool(opts.get("doArdl", True)),
                do_johansen=bool(opts.get("doJohansen", True)),
                do_eg=bool(opts.get("doEngleGranger", True)))
        except Exception as exc:
            res["errors"]["cointegration"] = str(exc)
            res["errors"]["cointegrationTrace"] = traceback.format_exc(limit=3)

    # ---------------- VAR ----------------
    if mods.get("var"):
        try:
            use_diff = opts.get("varUseDiff")
            if use_diff is None:
                # tum seriler I(1) ve esbutunlesme yoksa fark al
                ci = res.get("cointegration") or {}
                if ci:
                    has_ci = bool((ci.get("overall") or {}).get("cointegrated"))
                else:
                    has_ci = bool(hints.get("cointegrated"))
                all_i1 = bool(orders) and all(v == "I(1)" for v in orders.values())
                use_diff = bool(all_i1 and not has_ci)
            res["var"] = varmod.run(
                ts, variables=variables,
                maxlags=int(opts.get("varMaxLags", 8)),
                ic=str(opts.get("varIC", "aic")),
                lags=opts.get("varLags"),
                irf_periods=int(opts.get("irfPeriods", 10)),
                use_diff=use_diff,
                dmax=int(opts.get("tyDmax", 1)))
        except Exception as exc:
            res["errors"]["var"] = str(exc)
            res["errors"]["varTrace"] = traceback.format_exc(limit=3)

    # ---------------- GARCH ----------------
    if mods.get("garch"):
        try:
            res["garch"] = garch_mod.run(
                ts, dep,
                models=tuple(opts.get("garchModels", ["GARCH", "GJR", "EGARCH"])),
                p=int(opts.get("garchP", 1)), q=int(opts.get("garchQ", 1)),
                dist=str(opts.get("garchDist", "normal")),
                use_returns=bool(opts.get("garchReturns", False)))
        except Exception as exc:
            res["errors"]["garch"] = str(exc)
            res["errors"]["garchTrace"] = traceback.format_exc(limit=3)

    _availability(res, ts, mods)
    res["narrative"] = build_narrative(res, ts)
    res["elapsed"] = round(time.time() - t0, 2)
    return safe(res)


def _availability(res, ts: TSData, mods):
    notes = []
    n = ts.n
    if mods.get("arima") and not res.get("arima"):
        notes.append(f"ARIMA/SARIMA modeli tahmin edilemedi: en az 15 gözlem "
                     f"gereklidir (mevcut: {n}).")
    ci = res.get("cointegration") or {}
    if mods.get("cointegration") and not ci.get("available", True):
        notes.append(ci.get("note", "Eşbütünleşme analizi yapılamadı."))
    elif mods.get("cointegration") and ci.get("available") and not ci.get("overall"):
        notes.append("Eşbütünleşme testleri kısmen üretilebildi; gözlem sayısı veya "
                     "seri sayısı sınırda olabilir.")
    vr = res.get("var") or {}
    if mods.get("var") and not vr.get("available", True):
        notes.append(vr.get("note", "VAR analizi yapılamadı."))
    if mods.get("garch"):
        g = res.get("garch") or {}
        if g.get("note") and not g.get("models"):
            notes.append(g["note"])
    if n < 30:
        notes.append(f"Gözlem sayısı düşüktür (n = {n}). Zaman serisi testlerinin "
                     "gücü sınırlıdır; sonuçlar dikkatle yorumlanmalıdır.")
    if ts.freq == "?" and mods.get("arima"):
        notes.append("Frekans belirlenemediği için mevsimsel (SARIMA) bileşen "
                     "denenmemiştir.")
    res["availability"] = notes


# ==========================================================================
# APA anlatimi
# ==========================================================================

def _fmt(v, nd=3):
    if v is None:
        return "—"
    try:
        return f"{float(v):.{nd}f}".replace(".", ",")
    except (TypeError, ValueError):
        return str(v)


def _p(v):
    if v is None:
        return "p = —"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "p = —"
    if x < 0.001:
        return "p < ,001"
    return "p = " + f"{x:.3f}".replace(".", ",").replace("0,", ",", 1)


def _sig(v, a=0.05):
    try:
        return v is not None and float(v) < a
    except (TypeError, ValueError):
        return False


def _liste(xs, son=" ve "):
    xs = [str(x) for x in xs]
    if not xs:
        return ""
    if len(xs) == 1:
        return xs[0]
    return ", ".join(xs[:-1]) + son + xs[-1]


def build_narrative(res, ts: TSData):
    n = {}
    m = res["meta"]
    dep = res["depVar"]
    variables = m["vars"]

    n["design"] = (
        f"Araştırmada {m['start']}–{m['end']} dönemini kapsayan "
        f"{m['freqLabel'].lower()} frekanslı {m['n']} gözlemlik zaman serisi verisi "
        f"kullanılmıştır. Analize konu olan seri"
        f"{'ler' if len(variables) > 1 else ''} {_liste(variables)} olarak "
        f"belirlenmiştir." +
        (f" Modellemede bağımlı seri {dep} olarak alınmıştır."
         if len(variables) > 1 else ""))

    # ---- duraganlik
    st = res.get("stationarity")
    if st and st.get("results"):
        i0 = [r["variable"] for r in st["results"] if r["order"] == "I(0)"]
        i1 = [r["variable"] for r in st["results"] if r["order"] == "I(1)"]
        i2 = [r["variable"] for r in st["results"] if r["order"].startswith("I(2)")]
        parts = [
            f"Serilerin durağanlık özellikleri {st['options']['trendLabel'].lower()} "
            f"model varsayımı altında Genişletilmiş Dickey-Fuller (ADF), "
            f"Phillips-Perron (PP) ve KPSS testleri ile sınanmış; yapısal kırılma "
            f"olasılığına karşı Zivot-Andrews (1992) içsel kırılmalı birim kök testi "
            f"de uygulanmıştır."]
        if i0:
            parts.append(f"{_liste(i0)} serisinin/serilerinin düzeyde durağan, "
                         f"yani I(0), olduğu belirlenmiştir.")
        if i1:
            parts.append(f"{_liste(i1)} serisinin/serilerinin düzeyde durağan "
                         f"olmadığı, birinci farkı alındığında durağan hâle geldiği, "
                         f"dolayısıyla I(1) özelliği taşıdığı tespit edilmiştir.")
        if i2:
            parts.append(f"{_liste(i2)} serisinin/serilerinin birinci farkında da "
                         f"durağanlaşmadığı görülmüştür.")
        brk = [(r["variable"], r["breakLabel"]) for r in st["results"]
               if r.get("breakLabel")]
        if brk:
            parts.append("Zivot-Andrews testi kapsamında belirlenen içsel kırılma "
                         "tarihleri " + _liste([f"{v} için {b}" for v, b in brk])
                         + " olarak tespit edilmiştir.")
        parts.append(st["note"])
        n["stationarity"] = " ".join(parts)

    # ---- ARIMA
    ar = res.get("arima")
    if ar:
        d = ar.get("diagnostics") or {}
        fit = ar.get("fit") or {}
        parts = [
            f"{ar['variable']} serisi için Box-Jenkins yaklaşımı çerçevesinde "
            f"otomatik model seçimi yapılmış; {ar['ic'].upper()} bilgi ölçütünü "
            f"en küçükleyen model {ar['model']} olarak belirlenmiştir "
            f"(AIC = {_fmt(fit.get('aic'), 2)}, BIC = {_fmt(fit.get('bic'), 2)})."]
        sig = [c for c in ar["coeffs"]
               if _sig(c.get("p")) and c["name"] not in ("sigma2",)]
        if sig:
            parts.append("Modelde istatistiksel olarak anlamlı bulunan parametreler "
                         + _liste([f"{c['name']} (β = {_fmt(c['B'])}, {_p(c['p'])})"
                                   for c in sig]) + " şeklindedir.")
        lb = d.get("ljungBox")
        if lb:
            parts.append(
                f"Artıkların ardışık bağımlılığı Ljung-Box testi ile sınanmış; "
                f"Q({lb['lag']}) = {_fmt(lb['stat'])}, {_p(lb['p'])} bulgusu "
                f"artıklarda otokorelasyon "
                f"{'bulunmadığını' if lb['ok'] else 'bulunduğunu'} göstermektedir.")
        jb = d.get("jarqueBera")
        if jb:
            parts.append(
                f"Jarque-Bera testi ({_fmt(jb['stat'])}, {_p(jb['p'])}) artıkların "
                f"normal dağılım varsayımını "
                f"{'karşıladığına' if jb['ok'] else 'karşılamadığına'} işaret etmektedir.")
        alm = d.get("archLM")
        if alm:
            parts.append(
                f"ARCH-LM testi ({_fmt(alm['stat'])}, {_p(alm['p'])}) sonucunda "
                f"artıklarda koşullu değişen varyans "
                f"{'tespit edilmemiştir' if alm['ok'] else 'tespit edilmiştir'}.")
        parts.append(
            "Modelin uyum ölçütleri RMSE = " + _fmt(fit.get("rmse")) +
            ", MAE = " + _fmt(fit.get("mae")) +
            (", MAPE = %" + _fmt(fit.get("mape"), 2) if fit.get("mape") is not None else "")
            + " olarak hesaplanmıştır.")
        fc = ar.get("forecast")
        if fc and fc.get("mean"):
            parts.append(
                f"Tahmin edilen model kullanılarak {fc['horizon']} dönemlik öngörü "
                f"üretilmiş; {fc['labels'][0]} dönemi için nokta tahmini "
                f"{_fmt(fc['mean'][0])} (%95 GA: {_fmt(fc['lower'][0])} – "
                f"{_fmt(fc['upper'][0])}), {fc['labels'][-1]} dönemi için "
                f"{_fmt(fc['mean'][-1])} (%95 GA: {_fmt(fc['lower'][-1])} – "
                f"{_fmt(fc['upper'][-1])}) olarak elde edilmiştir.")
        n["arima"] = " ".join(parts)

    # ---- esbutunlesme
    ci = res.get("cointegration")
    if ci and ci.get("available"):
        parts = ["Seriler arasındaki uzun dönemli ilişkinin varlığı araştırılmıştır."]
        eg = ci.get("engleGranger")
        if eg:
            parts.append(
                f"Engle-Granger iki aşamalı yöntemine göre eşbütünleşme regresyonu "
                f"artıklarına uygulanan ADF testi {_fmt(eg['stat'])} ({_p(eg['p'])}) "
                f"sonucunu vermiş; buna göre "
                f"{'eşbütünleşme ilişkisi bulunmaktadır' if eg['sig'] else 'eşbütünleşme ilişkisi bulunmamaktadır'}.")
        jo = ci.get("johansen")
        if jo:
            parts.append(
                f"Johansen testinde iz (trace) istatistiği {jo['rankTrace']}, "
                f"maksimum özdeğer istatistiği {jo['rankMaxEigen']} adet "
                f"eşbütünleşme vektörüne işaret etmiş; %5 anlamlılık düzeyinde "
                f"{jo['decision'].lower()}.")
        ve = ci.get("vecm")
        if ve:
            parts.append(
                f"Vektör hata düzeltme modelinde {ve['vars'][0]} denklemine ait "
                f"hata düzeltme katsayısı α = {_fmt(ve['speed'])} ({_p(ve['speedP'])}) "
                + ("olarak hesaplanmış; katsayının negatif ve anlamlı olması kısa "
                   "dönemli sapmaların uzun dönem dengesine doğru düzeltildiğini "
                   "göstermektedir." if ve["valid"] else
                   "olarak hesaplanmıştır; katsayı beklenen işaret ve anlamlılığı "
                   "taşımamaktadır."))
            if ve.get("halfLife"):
                parts.append(f"Dengeden sapmaların yarılanma süresi yaklaşık "
                             f"{_fmt(ve['halfLife'], 1)} dönemdir.")
        ard = ci.get("ardl")
        if ard:
            parts.append(
                f"{ard['orderLabel']} modeli üzerinden yürütülen sınır testinde "
                f"F-istatistiği {_fmt(ard['stat'])} olarak hesaplanmış; bu değer "
                f"%5 anlamlılık düzeyindeki alt sınır ({_fmt(ard['lower95'])}) ve "
                f"üst sınır ({_fmt(ard['upper95'])}) ile karşılaştırıldığında "
                f"{ard['decision'].lower()} sonucuna ulaşılmıştır.")
            lr = [c for c in (ard.get("longRun") or [])
                  if c["name"] != "Sabit" and _sig(c.get("p"))]
            if lr:
                parts.append(
                    "Uzun dönem katsayılarına göre "
                    + _liste([f"{c['name']} (β = {_fmt(c['B'])}, {_p(c['p'])})"
                              for c in lr])
                    + f" değişkenlerinin {ard['dep']} üzerinde uzun dönemde "
                    "istatistiksel olarak anlamlı etkisi bulunmaktadır.")
            if ard.get("ect") and ard.get("valid"):
                parts.append(
                    f"Hata düzeltme terimi {_fmt(ard['ect']['B'])} "
                    f"({_p(ard['ect']['p'])}) olarak negatif ve anlamlı bulunmuş; "
                    f"bu bulgu modelin uzun dönem dengesine yakınsadığını "
                    f"doğrulamaktadır.")
        if ci.get("overall"):
            parts.append(ci["overall"]["text"])
        n["cointegration"] = " ".join(parts)

    # ---- VAR
    vr = res.get("var")
    if vr and vr.get("available"):
        parts = [
            f"Değişkenler arasındaki dinamik etkileşim {'birinci farkları alınmış ' if vr['useDiff'] else ''}"
            f"seriler üzerinden kurulan VAR modeli ile incelenmiştir. "
            f"{vr['icLabel']} bilgi ölçütüne göre uygun gecikme uzunluğu "
            f"{vr['lags']} olarak belirlenmiştir."]
        stb = vr.get("stability") or {}
        if stb.get("stable") is not None:
            parts.append(stb["note"])
        d = vr.get("diagnostics") or {}
        if d.get("whiteness"):
            w = d["whiteness"]
            parts.append(
                f"Artıkların otokorelasyon içermediği portmanteau testi ile "
                f"sınanmış ({_fmt(w['stat'])}, {_p(w['p'])}); artıklarda "
                f"otokorelasyon {'bulunmamaktadır' if w['ok'] else 'bulunmaktadır'}.")
        sums = vr.get("todaYamamotoSummary") or vr.get("grangerSummary") or []
        method = ("Toda-Yamamoto (1995) gecikme genişletmeli"
                  if vr.get("todaYamamotoSummary") else "Granger")
        rel = [s for s in sums if s["relation"] != "Nedensellik yok"]
        if rel:
            parts.append(f"{method} nedensellik testi sonuçlarına göre "
                         + _liste([f"{s['pair']} arasında {s['relation'].lower()} "
                                   f"({s['arrow']})" for s in rel])
                         + " tespit edilmiştir.")
        none_rel = [s["pair"] for s in sums if s["relation"] == "Nedensellik yok"]
        if none_rel:
            parts.append(f"{_liste(none_rel)} çift/çiftleri arasında ise anlamlı bir "
                         f"nedensellik ilişkisine rastlanmamıştır.")
        fe = vr.get("fevd")
        if fe and fe.get("tables"):
            t0 = fe["tables"][0]
            last = t0["rows"][-1]
            own = last["shares"][0] if last["shares"] else None
            parts.append(
                f"Varyans ayrıştırması sonuçlarına göre {t0['variable']} "
                f"değişkenindeki değişimin {fe['periods']}. dönem sonunda "
                f"%{_fmt(own, 1)}'lik kısmı kendi şoklarından kaynaklanmaktadır.")
        n["var"] = " ".join(parts)

    # ---- GARCH
    g = res.get("garch")
    if g:
        parts = []
        a = g.get("archLM")
        if a:
            parts.append(
                f"{g['seriesLabel']} için Engle (1982) ARCH-LM testi "
                f"{_fmt(a['stat'])} ({_p(a['p'])}) sonucunu vermiş; "
                f"{a['decision'].lower()}.")
        best = g.get("best")
        if best:
            parts.append(
                f"AIC ölçütüne göre en uygun volatilite modeli {best['model']} "
                f"olarak belirlenmiştir (AIC = {_fmt(best['fit']['aic'], 2)}). "
                f"Modelin kalıcılık (persistence) katsayısı "
                f"{_fmt(best['persistence'])} olup, bu değerin 1'in "
                f"{'altında olması volatilite şoklarının zamanla söndüğünü' if best['stationary'] else 'üzerinde olması volatilite sürecinin patlayıcı olduğunu'} "
                f"göstermektedir.")
            if best.get("halfLife"):
                parts.append(f"Volatilite şoklarının yarılanma süresi yaklaşık "
                             f"{_fmt(best['halfLife'], 1)} dönemdir.")
            pa = best.get("postArch")
            if pa:
                parts.append(
                    f"Model sonrası standartlaştırılmış artıklara uygulanan ARCH-LM "
                    f"testi ({_p(pa['p'])}) koşullu değişen varyansın "
                    f"{'giderildiğini' if not pa['sig'] else 'tam olarak giderilemediğini'} "
                    f"göstermektedir.")
        elif g.get("note"):
            parts.append(g["note"])
        if parts:
            n["garch"] = " ".join(parts)

    libs = ", ".join(ENGINE["libraries"])
    n["method"] = (
        "Bu araştırmadaki zaman serisi analizleri, web tabanlı AIS Akademi Zaman "
        "Serisi Analizi Modülü kullanılarak gerçekleştirilmiştir. Birim kök testleri, "
        "Box-Jenkins model seçimi, eşbütünleşme sınamaları, VAR/VECM tahminleri ile "
        f"volatilite modellemesi Python 3.11 programlama dili ve {libs} "
        "kütüphaneleri (Seabold & Perktold, 2010; Sheppard vd., 2024) ile bulut "
        "sunucu tabanlı olarak yürütülmüştür.")
    return n
