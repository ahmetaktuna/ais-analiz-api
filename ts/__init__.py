"""
AIS Akademi - Zaman Serisi Analizi Motoru
=========================================
Tek degiskenli ve cok degiskenli zaman serisi ekonometrisi cekirdegi.

Moduller
--------
prepare      : Zaman sutunu tespiti, frekans cikarimi, tanimlayici istatistikler
stationarity : ADF, PP, KPSS ve Zivot-Andrews yapisal kirilmali birim kok testleri
arima        : ACF/PACF, otomatik ARIMA/SARIMA secimi, artik tanilari, ongoru
coint        : Engle-Granger, Johansen, VECM ve ARDL sinir testi
varmod       : VAR gecikme secimi, Granger ve Toda-Yamamoto nedensellik,
               etki-tepki fonksiyonlari, varyans ayristirmasi
garch        : ARCH-LM testi ve GARCH ailesi modelleri
engine       : Tum modulleri calistiran ana orkestrasyon ve APA anlatimi

Not: Ortak matematiksel yardimcilar panel.utils modulunden kullanilir; bu
paket panel/ paketiyle birlikte dagitilir.
"""

__version__ = "1.0.0"
__all__ = ["engine"]
