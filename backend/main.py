from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from analyzer import analyze_audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("chordgrid.api")

app = FastAPI(title="ChordGrid API", version="0.2.0")

# V0.2 prototype: allow the Render static frontend and local development.
# Tighten this to the exact frontend URL once the product is stabilized.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "40"))


@app.get("/")
def root() -> dict:
    return {"name": "ChordGrid API", "version": "0.2.0", "status": "ok"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.2.0"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), mode: str = Form("standard")) -> dict:
    filename = file.filename or "audio"
    suffix = Path(filename).suffix.lower()
    logger.info("request_analyze file=%s mode=%s", filename, mode)

    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"Format non pris en charge. Formats: {', '.join(sorted(ALLOWED_SUFFIXES))}")

    raw = await file.read((MAX_MB * 1024 * 1024) + 1)
    if len(raw) > MAX_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Fichier > {MAX_MB} Mo")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        result = analyze_audio(tmp_path, filename=filename, mode=mode)
        logger.info("request_complete file=%s processing=%.2fs", filename, result.get("processingSeconds", -1))
        return result
    except Exception as exc:
        logger.exception("request_failed file=%s", filename)
        raise HTTPException(status_code=422, detail=f"Analyse impossible: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
