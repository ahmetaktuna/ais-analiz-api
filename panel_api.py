"""
AIS Akademi - Panel Veri Analizi API
====================================
Bu dosya HEM tek basina calisabilir HEM de mevcut API'nize eklenebilir.

A) TEK BASINA (FastAPI):
   uvicorn panel_api:app --host 0.0.0.0 --port $PORT

B) MEVCUT FastAPI UYGULAMANIZA EKLEME (onerilen):
   # app.py icinde
   from panel_api import router as panel_router
   app.include_router(panel_router)

C) MEVCUT FLASK UYGULAMANIZA EKLEME:
   # app.py icinde
   from panel_api import flask_blueprint
   app.register_blueprint(flask_blueprint)

Uc noktalar:
   POST /panel-analyze  -> tam analiz
   POST /panel-detect   -> sadece panel yapisi tespiti (hizli on kontrol)
   GET  /panel-health   -> saglik kontrolu
"""
from __future__ import annotations

import logging
import os
import traceback

from panel import engine
from panel.engine import AnalysisError
from panel.prepare import detect_structure

log = logging.getLogger("panel-api")

MAX_PAYLOAD_ROWS = int(os.getenv("PANEL_MAX_ROWS", "200000"))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "PANEL_CORS_ORIGINS",
    "https://www.aisakademi.com,https://aisakademi.com,http://localhost:3000"
).split(",") if o.strip()]


# --------------------------------------------------------------------------
# Cekirdek islem (framework bagimsiz)
# --------------------------------------------------------------------------

def handle_analyze(payload: dict):
    """Donen: (http_status, body_dict)"""
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
        log.exception("panel analyze failed")
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
        det = detect_structure(df)
        out = {"ok": True, "detection": det, "columns": list(df.columns)}
        if det["idVar"] and det["timeVar"]:
            g = df.groupby(det["idVar"])[det["timeVar"]].count()
            out["summary"] = {
                "N": int(df[det["idVar"]].nunique()),
                "T": int(df[det["timeVar"]].nunique()),
                "nobs": int(len(df)),
                "balanced": bool(g.nunique() == 1),
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

    router = APIRouter(tags=["panel"])

    @router.get("/panel-health")
    async def panel_health():
        return {"ok": True, "service": "panel", "version": engine.ENGINE["version"]}

    @router.post("/panel-detect")
    async def panel_detect(request: Request):
        body = await request.json()
        status, out = handle_detect(body)
        return JSONResponse(status_code=status, content=out)

    @router.post("/panel-analyze")
    async def panel_analyze(request: Request):
        body = await request.json()
        status, out = handle_analyze(body)
        return JSONResponse(status_code=status, content=out)

    def create_app():
        """Tek basina calisan FastAPI uygulamasi olusturur."""
        _app = FastAPI(title="AIS Akademi Panel Veri Analizi API",
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
            return {"ok": True, "endpoints": ["/panel-analyze", "/panel-detect",
                                              "/panel-health"]}
        return _app

    def __getattr__(name):
        """`uvicorn panel_api:app` calisir; sadece router import edildiginde
        ikinci bir FastAPI uygulamasi olusturulmaz (PEP 562)."""
        if name == "app":
            globals()["app"] = create_app()
            return globals()["app"]
        raise AttributeError(f"module 'panel_api' has no attribute {name!r}")
except ImportError:  # pragma: no cover
    router = None
    create_app = None


# --------------------------------------------------------------------------
# Flask
# --------------------------------------------------------------------------
try:
    from flask import Blueprint, jsonify, request

    flask_blueprint = Blueprint("panel", __name__)

    @flask_blueprint.route("/panel-health", methods=["GET"])
    def _fl_health():
        return jsonify({"ok": True, "service": "panel",
                        "version": engine.ENGINE["version"]})

    @flask_blueprint.route("/panel-detect", methods=["POST", "OPTIONS"])
    def _fl_detect():
        if request.method == "OPTIONS":
            return ("", 204)
        status, out = handle_detect(request.get_json(force=True, silent=True) or {})
        return jsonify(out), status

    @flask_blueprint.route("/panel-analyze", methods=["POST", "OPTIONS"])
    def _fl_analyze():
        if request.method == "OPTIONS":
            return ("", 204)
        status, out = handle_analyze(request.get_json(force=True, silent=True) or {})
        return jsonify(out), status
except ImportError:  # pragma: no cover
    flask_blueprint = None


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("panel_api:app", host="0.0.0.0",
                port=int(os.getenv("PORT", "8000")), reload=False)
