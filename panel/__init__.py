"""
AIS Akademi - Panel Veri Analizi Motoru
=======================================
Panel (boylamsal) veri setleri icin kapsamli ekonometrik analiz cekirdegi.

Moduller
--------
utils      : Ortak matematiksel yardimcilar (OLS, ADF, uzun donem kovaryans)
prepare    : Veri hazirlama, panel yapisi tespiti, tanimlayici istatistikler
models     : Havuzlanmis EKK, Sabit Etkiler, Rastgele Etkiler, Hausman, LM, F
diagnostics: Degisen varyans, otokorelasyon, yatay kesit bagimliligi
unitroot   : LLC, IPS, Fisher-ADF/PP, Hadri panel birim kok testleri
coint      : Pedroni, Kao, Westerlund panel esbutunlesme testleri
causality  : Dumitrescu-Hurlin ve havuzlanmis Granger nedensellik
dynamic    : Arellano-Bond / Blundell-Bond GMM ve Panel ARDL (MG/PMG/DFE)
engine     : Tum modulleri calistiran ana orkestrasyon fonksiyonu
"""

__version__ = "1.0.0"
__all__ = ["engine"]
