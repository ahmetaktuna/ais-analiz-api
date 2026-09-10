"""Ana orkestrasyon: istek sozlugu -> tam analiz sonucu."""
from __future__ import annotations

import time
import traceback

import numpy as np
import pandas as pd

from . import causality, coint, diagnostics, dynamic, models, unitroot
from .prepare import MAX_ROWS, MAX_UNITS, Panel, correlation, descriptives, detect_structure
from .utils import f, safe

ENGINE = {
    "name": "AIS Akademi Panel Veri Analizi Motoru",
    "version": "1.0.0",
    "python": "3.11",
    "libraries": ["NumPy", "pandas", "SciPy", "statsmodels", "linearmodels"],
}

DEFAULT_MODULES = {
    "descriptives": True,
    "basic": True,
    "diagnostics": True,
    "unitroot": False,
    "cointegration": False,
    "causality": False,
    "dynamic": False,
}

MC_PRESETS = {"fast": 150, "standard": 300, "precise": 800}


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


def analyze(payload: dict) -> dict:
    t0 = time.time()
    df = _to_frame(payload.get("data"))
    if len(df) > MAX_ROWS:
        raise AnalysisError(f"Veri seti çok büyük (>{MAX_ROWS:,} satır).")

    id_var = payload.get("idVar")
    time_var = payload.get("timeVar")
    detected = detect_structure(df)
    if not id_var:
        id_var = detected["idVar"]
    if not time_var:
        time_var = detected["timeVar"]
    if not id_var or not time_var:
        raise AnalysisError("Birim (id) ve zaman değişkenleri belirlenemedi.")
    for c in (id_var, time_var):
        if c not in df.columns:
            raise AnalysisError(f"'{c}' sütunu veri setinde bulunamadı.")

    dep = payload.get("depVar")
    indep = list(payload.get("indepVars") or [])
    if not dep:
        raise AnalysisError("Bağımlı değişken seçilmedi.")
    if not indep:
        raise AnalysisError("En az bir bağımsız değişken seçilmelidir.")
    if dep in indep:
        raise AnalysisError("Bağımlı değişken aynı zamanda bağımsız değişken olamaz.")
    if dep in (id_var, time_var):
        raise AnalysisError("Bağımlı değişken, birim veya zaman sütunu olamaz.")
    if id_var in indep:
        raise AnalysisError("Birim (id) sütunu bağımsız değişken olarak kullanılamaz.")
    indep = list(dict.fromkeys(indep))
    missing = [c for c in [dep] + indep if c not in df.columns]
    if missing:
        raise AnalysisError("Şu sütunlar bulunamadı: " + ", ".join(missing))

    pnl = Panel(df, id_var, time_var, dep, indep)
    if pnl.N < 2:
        raise AnalysisError("Panel analizi için en az 2 birim (kesit) gereklidir.")
    if pnl.N > MAX_UNITS:
        raise AnalysisError(f"Birim sayısı çok yüksek (>{MAX_UNITS}).")
    if pnl.T < 2:
        raise AnalysisError("Panel analizi için en az 2 zaman noktası gereklidir.")
    if pnl.t_max < 2:
        raise AnalysisError(
            "Her birim için yalnızca tek gözlem bulunuyor; bu bir yatay kesit "
            "(cross-section) veri setidir, panel veri seti değildir. Panel analizi "
            "için her birimin en az 2 döneme ait gözlemi olmalıdır. Birim ve zaman "
            "sütunlarını kontrol ediniz.")
    if pnl.nobs < len(indep) + 5:
        raise AnalysisError("Gözlem sayısı model için yetersiz.")

    mods = dict(DEFAULT_MODULES)
    mods.update(payload.get("modules") or {})
    opts = payload.get("options") or {}
    mc = MC_PRESETS.get(str(opts.get("precision", "standard")), 300)

    res = {
        "ok": True,
        "engine": ENGINE,
        "meta": pnl.info(),
        "detection": detected,
        "modules": mods,
        "options": dict(opts),
        "warnings": [],
        "errors": {},
    }
    if pnl.dropped_na:
        res["warnings"].append(
            f"{pnl.dropped_na} satır eksik veri (NaN) nedeniyle analiz dışı bırakıldı.")
    if pnl.dropped_dup:
        res["warnings"].append(
            f"{pnl.dropped_dup} yinelenen birim-zaman kaydı çıkarıldı.")
    if not pnl.balanced:
        res["warnings"].append(
            f"Panel dengesizdir (birim başına {pnl.t_min}–{pnl.t_max} gözlem). "
            "Bazı testler dengeli alt panel üzerinde hesaplanmıştır.")

    # ---------------- Tanimlayici ----------------
    if mods.get("descriptives", True):
        try:
            res["descriptives"] = descriptives(pnl)
            res["correlation"] = correlation(pnl)
        except Exception as exc:
            res["errors"]["descriptives"] = str(exc)

    # ---------------- Temel modeller ----------------
    fit_objs = None
    if mods.get("basic", True):
        try:
            effects = opts.get("effects", "entity")
            cov = opts.get("covType", "clustered")
            basic, fit_objs = models.fit_all(pnl, effects=effects, cov_type=cov)
            res["basic"] = basic
            if opts.get("robustComparison", True):
                res["basic"]["seComparison"] = models.robust_variants(pnl, effects)
        except Exception as exc:
            res["errors"]["basic"] = str(exc)
            res["errors"]["basicTrace"] = traceback.format_exc(limit=3)

    # ---------------- Tanilayici testler ----------------
    if mods.get("diagnostics", True) and fit_objs:
        try:
            fe_r = np.asarray(fit_objs["fe"].resids.values, float).ravel()
            po_r = np.asarray(fit_objs["pooled"].resids.values, float).ravel()
            res["diagnostics"] = diagnostics.run_all(pnl, fe_r, po_r)
        except Exception as exc:
            res["errors"]["diagnostics"] = str(exc)

    # ---------------- Panel birim kok ----------------
    if mods.get("unitroot"):
        try:
            src = pnl if pnl.balanced else pnl.balanced_subset()
            res["unitRoot"] = unitroot.run_all(
                src,
                trend=opts.get("urTrend", "c"),
                lags=opts.get("urLags", "auto"),
                maxlag=int(opts.get("urMaxLag", 2)),
                mc_reps=mc,
                tests=tuple(opts.get("urTests",
                                     ["llc", "ips", "fisher_adf", "fisher_pp", "hadri"])),
            )
            res["unitRoot"]["usedBalancedSubset"] = bool(src is not pnl)
        except Exception as exc:
            res["errors"]["unitRoot"] = str(exc)

    # ---------------- Esbutunlesme ----------------
    if mods.get("cointegration"):
        try:
            src = pnl if pnl.balanced else pnl.balanced_subset()
            res["cointegration"] = coint.run_all(
                src, det=opts.get("cointDet", "c"),
                lags=int(opts.get("cointLags", 1)),
                reps=mc, longrun=bool(opts.get("longRun", True)))
            res["cointegration"]["usedBalancedSubset"] = bool(src is not pnl)
        except Exception as exc:
            res["errors"]["cointegration"] = str(exc)

    # ---------------- Nedensellik ----------------
    if mods.get("causality"):
        try:
            res["causality"] = causality.run_all(
                pnl, lags=int(opts.get("causalityLags", 1)),
                include_pooled=bool(opts.get("pooledGranger", True)))
        except Exception as exc:
            res["errors"]["causality"] = str(exc)

    # ---------------- Dinamik panel ----------------
    if mods.get("dynamic"):
        try:
            res["dynamic"] = dynamic.run_all(
                pnl,
                gmm=bool(opts.get("gmm", True)),
                ardl=bool(opts.get("ardl", True)),
                ylag=int(opts.get("gmmYLag", 1)),
                max_inst=int(opts.get("gmmMaxInstruments", 3)),
                twostep=bool(opts.get("gmmTwoStep", True)),
                time_dummies=bool(opts.get("gmmTimeDummies", False)),
                collapse=bool(opts.get("gmmCollapse", True)),
                p=int(opts.get("ardlP", 1)), q=int(opts.get("ardlQ", 1)))
        except Exception as exc:
            res["errors"]["dynamic"] = str(exc)

    _availability_notes(res, pnl, mods)
    res["narrative"] = build_narrative(res, pnl)
    res["elapsed"] = round(time.time() - t0, 2)
    return safe(res)


def _availability_notes(res, pnl: Panel, mods):
    """Istenen ama uretilemeyen modulller icin aciklayici not ekler."""
    notes = []
    T, N = pnl.t_min, pnl.N

    if mods.get("unitroot") and not (res.get("unitRoot") or {}).get("results"):
        notes.append(
            f"Panel birim kök testleri hesaplanamadı: bu testler birim başına en az "
            f"8–10 dönem gerektirir (mevcut en kısa seri {T} dönem). Zaman boyutu "
            f"kısa panellerde (mikro panel) durağanlık analizi anlamlı değildir.")
    ci = res.get("cointegration") or {}
    if mods.get("cointegration") and not ci.get("overall"):
        notes.append(
            f"Panel eşbütünleşme testleri hesaplanamadı veya kısmen üretildi: "
            f"bu testler birim başına en az 10–12 dönem gerektirir "
            f"(mevcut en kısa seri {T} dönem).")
    if mods.get("causality") and not (res.get("causality") or {}).get("pairs"):
        notes.append(
            f"Dumitrescu-Hurlin nedensellik testi hesaplanamadı: birim başına "
            f"en az 2K + 5 dönem gereklidir (mevcut en kısa seri {T} dönem).")
    dyn = res.get("dynamic") or {}
    ardl = dyn.get("ardl") or {}
    if mods.get("dynamic") and dyn and not any(ardl.get(k) for k in ("mg", "pmg", "dfe")):
        notes.append(
            f"Panel ARDL (MG/PMG/DFE) tahmincileri hesaplanamadı: birim başına "
            f"yeterli dönem bulunmuyor (mevcut en kısa seri {T} dönem). "
            f"Kısa panellerde Arellano-Bond GMM tahmincisi tercih edilmelidir.")
    if mods.get("dynamic") and dyn and dyn.get("gmm") is None:
        notes.append("Arellano-Bond GMM tahmin edilemedi: dinamik model için "
                     "birim başına en az 4 dönem gereklidir.")
    if N < 5:
        notes.append(f"Birim sayısı çok düşüktür (N = {N}). Panel tahmincilerinin "
                     "asimptotik özellikleri N büyüdükçe geçerli olduğundan "
                     "sonuçlar dikkatle yorumlanmalıdır.")
    if T < 5 and mods.get("basic"):
        notes.append(f"Zaman boyutu kısadır (T = {T}). Sabit etkiler tahmincisinde "
                     "Nickell yanlılığı riski bulunmaktadır.")
    res["availability"] = notes


# ==========================================================================
# APA anlatimi
# ==========================================================================

def _fmt(v, nd=3):
    if v is None:
        return "—"
    try:
        s = f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)
    return s.replace(".", ",")


def _p(v):
    """APA usulu p-degeri (Turkce ondalik ayraci ile)."""
    if v is None:
        return "p = —"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "p = —"
    if x < 0.001:
        return "p < ,001"
    return "p = " + f"{x:.3f}".replace(".", ",").replace("0,", ",", 1)


def _sig(v, alpha=0.05):
    """p-degeri anlamli mi? (0.0 degerinin falsy olmasina karsi guvenli)"""
    try:
        return v is not None and float(v) < alpha
    except (TypeError, ValueError):
        return False


def _liste(xs, son=" ve "):
    xs = list(xs)
    if not xs:
        return ""
    if len(xs) == 1:
        return xs[0]
    return ", ".join(xs[:-1]) + son + xs[-1]


def build_narrative(res, pnl: Panel):
    """APA formatinda Turkce yorum paragraflari uretir."""
    n = {}
    m = res["meta"]
    dep, indep = m["depVar"], m["indepVars"]
    ind_txt = _liste(indep)

    n["design"] = (
        f"Araştırmada {m['N']} birim (kesit) ve {m['T']} döneme ait toplam "
        f"{m['nobs']} gözlemden oluşan "
        f"{'dengeli' if m['balanced'] else 'dengesiz'} panel veri seti kullanılmıştır. "
        f"Analiz döneminin {m['timeStart']}–{m['timeEnd']} aralığını kapsadığı, "
        f"birim başına gözlem sayısının {m['tMin']} ile {m['tMax']} arasında değiştiği "
        f"görülmektedir. Araştırmanın bağımlı değişkeni {dep}, bağımsız "
        f"değişken{'i' if len(indep) == 1 else 'leri'} ise {ind_txt} olarak "
        f"belirlenmiştir."
    )

    # ---- temel model
    b = res.get("basic")
    if b:
        sel = b["selection"]
        fin = b["final"]
        parts = []
        if b.get("fTest"):
            ft = b["fTest"]
            parts.append(
                f"Havuzlanmış EKK ile Sabit Etkiler modelleri arasında tercih "
                f"yapabilmek için uygulanan F testi sonucunda "
                f"F({ft['df1']}, {ft['df2']}) = {_fmt(ft['stat'])}, "
                f"{_p(ft['p'])} bulunmuştur; bu bulgu birim etkilerinin "
                f"{'istatistiksel olarak anlamlı olduğunu' if _sig(ft['p']) else 'anlamlı olmadığını'} "
                f"göstermektedir.")
        if b.get("lmTest"):
            lt = b["lmTest"]
            parts.append(
                f"Breusch-Pagan Lagrange çarpanı testi sonucunda "
                f"χ²(1) = {_fmt(lt['stat'])}, {_p(lt['p'])} elde edilmiş; "
                f"birim etkilerinin varyansının sıfır olduğu yönündeki sıfır hipotezi "
                f"{'reddedilmiştir' if _sig(lt['p']) else 'reddedilememiştir'}.")
        if b.get("hausman"):
            hs = b["hausman"]
            parts.append(
                f"Sabit Etkiler ile Rastgele Etkiler modelleri arasındaki tercihi "
                f"belirlemek üzere uygulanan Hausman testi χ²({hs['df']}) = "
                f"{_fmt(hs['stat'])}, {_p(hs['p'])} sonucunu vermiştir. "
                f"Buna göre {hs['decision']} modeli tercih edilmiştir.")
        parts.append(sel["reason"])
        n["modelSelection"] = " ".join(parts)

        sig = [c for c in fin["coeffs"]
               if c["name"] != "Sabit" and c["p"] is not None and c["p"] < 0.05]
        insig = [c for c in fin["coeffs"]
                 if c["name"] != "Sabit" and (c["p"] is None or c["p"] >= 0.05)]
        key = sel["preferred"]
        if key == "fe":
            r2, r2lab = fin.get("r2Within"), "birim içi (within) R²"
        elif key == "re":
            r2, r2lab = (fin.get("r2Overall") if fin.get("r2Overall") is not None
                         else fin.get("r2")), "genel (overall) R²"
        else:
            r2, r2lab = fin.get("r2"), "R²"
        lines = [
            f"{ind_txt} değişken{'inin' if len(indep) == 1 else 'lerinin'} {dep} "
            f"üzerindeki etkisini incelemek amacıyla tahmin edilen "
            f"{sel['preferredLabel']} modeline ilişkin sonuçlar aşağıda sunulmuştur. "
            f"Model {b.get('covLabel', '')} ile tahmin edilmiştir. "
            f"Modelin açıklayıcılık gücü {r2lab} = {_fmt(r2)} olarak hesaplanmıştır; "
            f"F({fin.get('dfModel')}, {fin.get('dfResid')}) = {_fmt(fin.get('F'))}, "
            f"{_p(fin.get('pF'))} bulgusu modelin bütün olarak istatistiksel açıdan "
            f"{'anlamlı' if _sig(fin.get('pF')) else 'anlamsız'} olduğunu göstermektedir."
        ]
        for c in fin["coeffs"]:
            if c["name"] == "Sabit":
                continue
            yon = "pozitif" if (c["B"] or 0) > 0 else "negatif"
            anlam = "anlamlı" if (c["p"] is not None and c["p"] < 0.05) else "anlamsız"
            lines.append(
                f"{c['name']} değişkeni için hesaplanan katsayı B = {_fmt(c['B'])} "
                f"(SH = {_fmt(c['SE'])}), t = {_fmt(c['t'])}, {_p(c['p'])} olup; "
                f"bu değişkenin {dep} üzerinde istatistiksel olarak {anlam}"
                f"{' ve ' + yon if anlam == 'anlamlı' else ''} bir etkisinin bulunduğu "
                f"görülmektedir."
                + (f" Buna göre {c['name']} değişkenindeki bir birimlik artış, "
                   f"{dep} değişkeninde ortalama {_fmt(abs(c['B']))} birimlik "
                   f"{'artışa' if (c['B'] or 0) > 0 else 'azalışa'} yol açmaktadır."
                   if anlam == "anlamlı" else ""))
        if sig:
            lines.append(
                f"Sonuç olarak {_liste([c['name'] for c in sig])} "
                f"değişken{'inin' if len(sig) == 1 else 'lerinin'} {dep} üzerinde "
                f"anlamlı bir yordayıcı etkisi bulunurken"
                + (f", {_liste([c['name'] for c in insig])} "
                   f"değişken{'inin' if len(insig) == 1 else 'lerinin'} anlamlı bir "
                   f"etkisine rastlanmamıştır." if insig else " modeldeki tüm "
                   "değişkenler anlamlı bulunmuştur."))
        else:
            lines.append(f"Modele dâhil edilen bağımsız değişkenlerin hiçbirinin {dep} "
                         f"üzerinde istatistiksel olarak anlamlı bir etkisi bulunmamıştır.")
        n["model"] = " ".join(lines)

    # ---- tanilayici
    d = res.get("diagnostics")
    if d:
        parts = []
        het = d["heteroskedasticity"].get("modifiedWald")
        if het:
            parts.append(
                f"Gruplar arası değişen varyansın sınanması amacıyla uygulanan "
                f"Modified Wald testi χ²({het['df']}) = {_fmt(het['stat'])}, "
                f"{_p(het['p'])} sonucunu vermiş; modelde değişen varyans sorununun "
                f"{'bulunduğu' if het['problem'] else 'bulunmadığı'} belirlenmiştir.")
        wl = d["autocorrelation"].get("wooldridge")
        if wl:
            parts.append(
                f"Otokorelasyonun test edilmesi amacıyla uygulanan Wooldridge testi "
                f"F({wl['df1']}, {wl['df2']}) = {_fmt(wl['stat'])}, {_p(wl['p'])} "
                f"sonucunu vermiş; birinci derece otokorelasyon sorununun "
                f"{'bulunduğu' if wl['problem'] else 'bulunmadığı'} tespit edilmiştir.")
        csd = d.get("crossSectionDependence")
        if csd:
            cd = csd["tests"][0]
            parts.append(
                f"Yatay kesit bağımlılığı Pesaran CD testi ile sınanmış; "
                f"CD = {_fmt(cd['stat'])}, {_p(cd['p'])} bulgusu birimler arasında "
                f"yatay kesit bağımlılığının "
                f"{'bulunduğuna' if cd['problem'] else 'bulunmadığına'} işaret etmektedir.")
        mc_ = d.get("multicollinearity")
        if mc_ and mc_.get("maxVif") is not None:
            parts.append(
                f"Çoklu doğrusal bağlantı açısından hesaplanan en yüksek VIF değeri "
                f"{_fmt(mc_['maxVif'])} olup, bu değerin 10 eşiğinin "
                f"{'üzerinde olması çoklu bağlantı sorununa' if mc_['problem'] else 'altında olması çoklu bağlantı sorunu bulunmadığına'} "
                f"işaret etmektedir.")
        if d.get("issues"):
            parts.append(
                f"Tespit edilen {_liste(d['issues'])} sorun"
                f"{'u' if len(d['issues']) == 1 else 'ları'} nedeniyle model "
                f"{d['recommendedSE']} ile yeniden tahmin edilmiştir.")
        else:
            parts.append("Tanılayıcı testler sonucunda modelin klasik varsayımları "
                         "karşıladığı görülmüştür.")
        n["diagnostics"] = " ".join(parts)

    # ---- birim kok
    ur = res.get("unitRoot")
    if ur and ur.get("results"):
        i1 = [r["variable"] for r in ur["results"] if r["order"] == "I(1)"]
        i0 = [r["variable"] for r in ur["results"] if r["order"] == "I(0)"]
        parts = [
            f"Serilerin durağanlık özellikleri {ur['options']['trendLabel'].lower()} "
            f"model varsayımı altında Levin-Lin-Chu (LLC), Im-Pesaran-Shin (IPS), "
            f"Fisher-ADF, Fisher-PP ve Hadri panel birim kök testleri ile sınanmıştır."]
        if i0:
            parts.append(f"{_liste(i0)} serisinin/serilerinin düzeyde durağan, "
                         f"yani I(0), olduğu belirlenmiştir.")
        if i1:
            parts.append(f"{_liste(i1)} serisinin/serilerinin düzeyde durağan olmadığı, "
                         f"birinci farkı alındığında durağan hâle geldiği, dolayısıyla "
                         f"I(1) özelliği taşıdığı tespit edilmiştir.")
        parts.append(ur["note"])
        n["unitRoot"] = " ".join(parts)

    # ---- esbutunlesme
    ci = res.get("cointegration")
    if ci and ci.get("overall"):
        o = ci["overall"]
        parts = [
            "Seriler arasındaki uzun dönemli ilişkinin varlığı Pedroni, Kao ve "
            "Westerlund panel eşbütünleşme testleri ile araştırılmıştır."]
        if ci.get("pedroni"):
            pd_ = ci["pedroni"]
            parts.append(f"Pedroni testinde hesaplanan 7 istatistikten "
                         f"{pd_['nSignificant']} tanesi %5 anlamlılık düzeyinde "
                         f"sıfır hipotezini reddetmiştir.")
        if ci.get("kao"):
            ka = ci["kao"]
            parts.append(f"Kao testi ADF = {_fmt(ka['stat'])}, {_p(ka['p'])} "
                         f"sonucunu vermiştir.")
        if ci.get("westerlund"):
            we = ci["westerlund"]
            parts.append(f"Westerlund hata düzeltme tabanlı testinde 4 istatistikten "
                         f"{we['nSignificant']} tanesi anlamlı bulunmuştur.")
        parts.append(o["text"] + " Buna göre değişkenler arasında uzun dönemli bir "
                     + ("eşbütünleşme ilişkisinin bulunduğu"
                        if o["cointegrated"] else "eşbütünleşme ilişkisinin bulunmadığı")
                     + " sonucuna ulaşılmıştır.")
        if ci.get("fmols"):
            cs = [c for c in ci["fmols"]["coeffs"] if c["p"] is not None and c["p"] < 0.05]
            if cs:
                parts.append(
                    "Grup ortalaması FMOLS tahmincisi ile elde edilen uzun dönem "
                    "katsayılarına göre " + _liste(
                        [f"{c['name']} (β = {_fmt(c['B'])}, {_p(c['p'])})" for c in cs])
                    + f" değişkenlerinin {dep} üzerinde uzun dönemde anlamlı etkisi "
                    "bulunmaktadır.")
        n["cointegration"] = " ".join(parts)

    # ---- nedensellik
    ca = res.get("causality")
    if ca and ca.get("summary"):
        parts = [f"Değişkenler arasındaki nedensellik ilişkileri, yatay kesit "
                 f"bağımlılığına ve heterojenliğe uygun olan Dumitrescu-Hurlin (2012) "
                 f"panel nedensellik testi ile {ca['lags']} gecikme uzunluğunda "
                 f"sınanmıştır."]
        for s in ca["summary"]:
            if s["relation"] == "Nedensellik yok":
                continue
            parts.append(f"{s['pair']} değişkenleri arasında {s['relation'].lower()} "
                         f"({s['arrow']}) tespit edilmiştir.")
        nones = [s["pair"] for s in ca["summary"] if s["relation"] == "Nedensellik yok"]
        if nones:
            parts.append(f"{_liste(nones)} değişken çiftleri arasında ise anlamlı bir "
                         f"nedensellik ilişkisine rastlanmamıştır.")
        n["causality"] = " ".join(parts)

    # ---- dinamik
    dy = res.get("dynamic")
    if dy:
        parts = []
        g = dy.get("gmm")
        if g:
            rho = g.get("persistence")
            parts.append(
                f"İçsellik sorununu dikkate almak amacıyla Arellano-Bond "
                f"fark GMM tahmincisi {g.get('stepShort', 'iki aşamalı')} "
                f"biçimde kullanılmıştır. "
                f"Bağımlı değişkenin gecikmeli değerine ait katsayı "
                f"{_fmt(rho)} olarak hesaplanmış; bu bulgu {dep} değişkeninin "
                f"{'yüksek' if abs(rho or 0) > 0.6 else 'orta' if abs(rho or 0) > 0.3 else 'düşük'} "
                f"düzeyde kalıcılık (persistence) gösterdiğine işaret etmektedir.")
            sg = g.get("hansen") or g.get("sargan")
            if sg and sg.get("p") is not None:
                parts.append(
                    f"Araç değişkenlerin geçerliliği {sg['name']} ile sınanmış; "
                    f"χ²({sg['df']}) = {_fmt(sg['stat'])}, {_p(sg['p'])} bulgusu "
                    f"araç değişkenlerin "
                    f"{'geçerli olduğunu' if sg['p'] > 0.05 else 'geçerliliğinin sorgulanması gerektiğini'} "
                    f"göstermektedir.")
            ar2 = next((a for a in g.get("ar", []) if a["order"] == 2), None)
            if ar2 and ar2.get("p") is not None:
                parts.append(
                    f"Arellano-Bond AR(2) testi z = {_fmt(ar2['stat'])}, "
                    f"{_p(ar2['p'])} sonucunu vermiş; ikinci derece otokorelasyonun "
                    f"{'bulunmadığı' if ar2['p'] > 0.05 else 'bulunduğu'} tespit edilmiştir.")
        ar = dy.get("ardl") or {}
        best = ar.get("pmg") or ar.get("mg")
        if best:
            hs = ar.get("hausman")
            if hs:
                parts.append(
                    f"MG ve PMG tahmincileri arasındaki tercihi belirlemek üzere "
                    f"uygulanan Hausman testi χ²({hs['df']}) = {_fmt(hs['stat'])}, "
                    f"{_p(hs['p'])} sonucunu vermiş ve {hs['decision']} tahmincisi "
                    f"tercih edilmiştir.")
            ec = best.get("ecm", {})
            if ec.get("B") is not None:
                parts.append(
                    f"{best['name']} ile elde edilen hata düzeltme katsayısı "
                    f"φ = {_fmt(ec['B'])} ({_p(ec.get('p'))}) olarak hesaplanmıştır. "
                    + ("Katsayının negatif ve istatistiksel olarak anlamlı olması, "
                       "kısa dönemde ortaya çıkan sapmaların uzun dönem dengesine "
                       "doğru düzeltildiğini göstermektedir."
                       if (ec['B'] < 0 and _sig(ec.get('p')))
                       else "Katsayının beklenen işaret ve anlamlılığı taşımaması, "
                            "uzun dönem denge ilişkisinin zayıf olduğuna işaret etmektedir."))
        if parts:
            n["dynamic"] = " ".join(parts)

    # ---- yontem ve atif
    libs = ", ".join(ENGINE["libraries"])
    n["method"] = (
        "Bu araştırmadaki panel veri analizleri, web tabanlı AIS Akademi Panel Veri "
        "Analizi Modülü kullanılarak gerçekleştirilmiştir. Model tahminleri, varsayım "
        "testleri, panel birim kök ve eşbütünleşme sınamaları ile nedensellik analizleri "
        f"Python 3.11 programlama dili ve {libs} kütüphaneleri "
        "(Seabold & Perktold, 2010; Sheppard vd., 2024) ile bulut sunucu tabanlı olarak "
        "yürütülmüştür. Simülasyon tabanlı olasılık değerleri, ilgili sıfır hipotezi "
        "altında aynı N ve T boyutlarında üretilen Monte Carlo dağılımlarından "
        "hesaplanmıştır.")
    return n
