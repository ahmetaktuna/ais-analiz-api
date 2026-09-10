"""
AIS Akademi - Zaman Serisi Analizi API
======================================
panel_api.py ile ayni desende calisir ve ayni uygulamaya eklenir.

A) MEVCUT FastAPI UYGULAMANIZA EKLEME (onerilen):
   # main.py icinde, panel router'inin hemen altina:
   from ts_api import router as ts_router
   app.include_router(ts_router)

B) FLASK:
   from ts_api import flask_blueprint as ts_blueprint
   app.register_blueprint(ts_blueprint)

C) TEK BASINA:
   uvicorn ts_api:app --host 0.0.0.0 --port $PORT

Uc noktalar:
   POST /ts-analyze  -> tam analiz
   POST /ts-detect   -> zaman sutunu ve frekans tespiti (hizli on kontrol)
   GET  /ts-health   -> saglik kontrolu

Not: Bu modul panel/ paketindeki ortak matematiksel yardimcilari kullanir,
bu nedenle panel/ klasoruyle birlikte dagitilmalidir.
"""
from __future__ import annotations

import logging
import os
import traceback

from ts import engine
from ts.engine import AnalysisError
from ts.prepare import TSData, detect_time_column

log = logging.getLogger("ts-api")

MAX_PAYLOAD_ROWS = int(os.getenv("TS_MAX_ROWS", "100000"))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "PANEL_CORS_ORIGINS",
    "https://www.aisakademi.com,https://aisakademi.com,http://localhost:3000"
).split(",") if o.strip()]


# --------------------------------------------------------------------------
# Cekirdek islem (framework bagimsiz)
# --------------------------------------------------------------------------

def handle_analyze(payload: dict):
    if not isinstance(payload, dict):
        return 400, {"ok": False, "error": "Geçersiz istek gövdesi."}
    data = payload.get("data")
    if not data:
        return 400, {"ok": False, "error": "Veri gönderilmedi."}
    if isinstance(data, list) and len(data) > MAX_PAYLOAD_ROWS:
        return 413, {"ok": False,
                     "error": f"Veri seti çok büyük (>{MAX_PAYLOAD_ROWS:,} satır)."}
    try:
        return 200, engine.analyze(payload)
    except AnalysisError as exc:
        return 400, {"ok": False, "error": str(exc)}
    except MemoryError:
        return 507, {"ok": False, "error": "Sunucu belleği yetersiz. "
                                           "Daha küçük bir veri seti deneyiniz."}
    except Exception as exc:  # pragma: no cover
        log.exception("ts analyze failed")
        return 500, {"ok": False,
                     "error": "Analiz sırasında beklenmeyen bir hata oluştu: "
                              f"{type(exc).__name__}: {exc}",
                     "trace": traceback.format_exc(limit=4)
                     if os.getenv("PANEL_DEBUG") else None}


def handle_detect(payload: dict):
    import pandas as pd
    data = (payload or {}).get("data")
    if not data:
        return 400, {"ok": False, "error": "Veri gönderilmedi."}
    try:
        df = pd.DataFrame(data)
        det = detect_time_column(df)
        out = {"ok": True, "detection": det, "columns": list(df.columns)}
        tv = (payload or {}).get("timeVar") or det.get("timeVar")
        if tv and tv in df.columns:
            others = [c for c in df.columns if c != tv]
            numeric = []
            for c in others:
                v = pd.to_numeric(df[c], errors="coerce")
                if v.notna().mean() >= 0.8:
                    numeric.append(c)
            if numeric:
                ts = TSData(df, tv, numeric[:1])
                out["summary"] = {
                    "timeVar": tv, "n": ts.n, "freq": ts.freq,
                    "freqLabel": ts.freq_label, "regular": ts.regular,
                    "start": ts.labels[0] if ts.labels else None,
                    "end": ts.labels[-1] if ts.labels else None,
                    "numericColumns": numeric,
                }
        return 200, out
    except Exception as exc:
        return 400, {"ok": False, "error": str(exc)}


# --------------------------------------------------------------------------
# FastAPI
# --------------------------------------------------------------------------
try:
    from fastapi import APIRouter, FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    router = APIRouter(tags=["zaman-serisi"])

    @router.get("/ts-health")
    async def ts_health():
        return {"ok": True, "service": "zaman-serisi",
                "version": engine.ENGINE["version"]}

    @router.post("/ts-detect")
    async def ts_detect(request: Request):
        status, out = handle_detect(await request.json())
        return JSONResponse(status_code=status, content=out)

    @router.post("/ts-analyze")
    async def ts_analyze(request: Request):
        status, out = handle_analyze(await request.json())
        return JSONResponse(status_code=status, content=out)

    def create_app():
        _app = FastAPI(title="AIS Akademi Zaman Serisi Analizi API",
                       version=engine.ENGINE["version"])
        _origin_regex = os.getenv(
            "PANEL_CORS_ORIGIN_REGEX",
            r"https?://(.*\.)?aisakademi\.com|http://(localhost|127\.0\.0\.1)(:\d+)?")
        _app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS or ["*"],
                            allow_origin_regex=_origin_regex,
                            allow_credentials=False, allow_methods=["*"],
                            allow_headers=["*"], max_age=86400)
        _app.include_router(router)

        @_app.get("/")
        async def root():
            return {"ok": True, "endpoints": ["/ts-analyze", "/ts-detect",
                                              "/ts-health"]}
        return _app

    def __getattr__(name):
        """`uvicorn ts_api:app` calisir; yalnizca router import edildiginde
        ikinci bir FastAPI uygulamasi olusturulmaz (PEP 562)."""
        if name == "app":
            globals()["app"] = create_app()
            return globals()["app"]
        raise AttributeError(f"module 'ts_api' has no attribute {name!r}")
except ImportError:  # pragma: no cover
    router = None
    create_app = None


# --------------------------------------------------------------------------
# Flask
# --------------------------------------------------------------------------
try:
    from flask import Blueprint, jsonify, request

    flask_blueprint = Blueprint("zaman_serisi", __name__)

    @flask_blueprint.route("/ts-health", methods=["GET"])
    def _fl_ts_health():
        return jsonify({"ok": True, "service": "zaman-serisi",
                        "version": engine.ENGINE["version"]})

    @flask_blueprint.route("/ts-detect", methods=["POST", "OPTIONS"])
    def _fl_ts_detect():
        if request.method == "OPTIONS":
            return ("", 204)
        status, out = handle_detect(request.get_json(force=True, silent=True) or {})
        return jsonify(out), status

    @flask_blueprint.route("/ts-analyze", methods=["POST", "OPTIONS"])
    def _fl_ts_analyze():
        if request.method == "OPTIONS":
            return ("", 204)
        status, out = handle_analyze(request.get_json(force=True, silent=True) or {})
        return jsonify(out), status
except ImportError:  # pragma: no cover
    flask_blueprint = None


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("ts_api:app", host="0.0.0.0",
                port=int(os.getenv("PORT", "8000")), reload=False)
