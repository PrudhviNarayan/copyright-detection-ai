from dataset.generate_dataset import generate_all
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
import shutil
import os
import uuid
from PIL import Image

from .database import get_db, init_db, CheckHistory
from .models.feature_extractors import (
    get_text_embedding,
    get_image_embedding,
    get_audio_embedding,
    get_video_embedding
)
from .faiss_index.indexer import get_indexer

app = FastAPI(title="Copyright Detection API")

# ✅ CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# temp folder
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)


# ✅ STARTUP
@app.on_event("startup")
def startup_event():
    init_db()
       # create dataset + FAISS index


# ✅ ROOT ROUTE
@app.get("/")
def root():
    return {"status": "Backend is running 🚀"}


# request model
class TextCheckRequest(BaseModel):
    text: str


# helpers
def save_upload_file(upload_file: UploadFile) -> str:
    ext = os.path.splitext(upload_file.filename)[1]
    filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(TEMP_DIR, filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return filepath


def determine_risk(score: float, threshold: float) -> str:
    return "HIGH" if score >= threshold else "LOW"


def log_history(db: Session, filename: str, file_type: str, score: float, risk: str, matched_with: str):
    history_entry = CheckHistory(
        filename=filename,
        file_type=file_type,
        similarity_score=score,
        risk_level=risk,
        matched_with=matched_with
    )
    db.add(history_entry)
    db.commit()


# ✅ SAFE SEARCH FUNCTION (MAIN FIX)
def safe_search(indexer, embedding):
    try:
        results = indexer.search(embedding, k=1)

        if not results or len(results) == 0:
            return 0.0, "No match found"

        item = results[0]

        # ensure correct format
        if isinstance(item, (list, tuple)) and len(item) == 2:
            meta, score = item
            filename = meta.get("filename", "Unknown")
            return float(score), filename

        return 0.0, "Invalid result"

    except Exception as e:
        print("SEARCH ERROR:", e)
        return 0.0, "Search failed"


# =========================
# TEXT
# =========================
@app.post("/check-text")
def check_text(request: TextCheckRequest, threshold: float = 0.8, db: Session = Depends(get_db)):
    try:
        embedding = get_text_embedding(request.text)
        indexer = get_indexer("text")

        score, matched_with = safe_search(indexer, embedding)

        risk = determine_risk(score, threshold)

        snippet = request.text[:30] + "..." if len(request.text) > 30 else request.text
        log_history(db, snippet, "text", score, risk, matched_with)

        return {
            "similarity_score": score,
            "top_match": matched_with,
            "risk_level": risk
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# IMAGE
# =========================
@app.post("/check-image")
def check_image(file: UploadFile = File(...), threshold: float = Form(0.8), db: Session = Depends(get_db)):
    try:
        filepath = save_upload_file(file)
        img = Image.open(filepath)

        embedding = get_image_embedding(img)
        indexer = get_indexer("image")

        score, matched_with = safe_search(indexer, embedding)

        risk = determine_risk(score, threshold)
        log_history(db, file.filename, "image", score, risk, matched_with)

        os.remove(filepath)

        return {
            "similarity_score": score,
            "top_match": matched_with,
            "risk_level": risk
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# AUDIO
# =========================
@app.post("/check-audio")
def check_audio(file: UploadFile = File(...), threshold: float = Form(0.8), db: Session = Depends(get_db)):
    try:
        filepath = save_upload_file(file)

        embedding = get_audio_embedding(filepath)
        indexer = get_indexer("audio")

        score, matched_with = safe_search(indexer, embedding)

        risk = determine_risk(score, threshold)
        log_history(db, file.filename, "audio", score, risk, matched_with)

        os.remove(filepath)

        return {
            "similarity_score": score,
            "top_match": matched_with,
            "risk_level": risk
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# VIDEO
# =========================
@app.post("/check-video")
def check_video(file: UploadFile = File(...), threshold: float = Form(0.8), db: Session = Depends(get_db)):
    try:
        filepath = save_upload_file(file)

        embedding = get_video_embedding(filepath)
        indexer = get_indexer("video")

        score, matched_with = safe_search(indexer, embedding)

        risk = determine_risk(score, threshold)
        log_history(db, file.filename, "video", score, risk, matched_with)

        os.remove(filepath)

        return {
            "similarity_score": score,
            "top_match": matched_with,
            "risk_level": risk
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# HISTORY
# =========================
@app.get("/history")
def get_history(limit: int = 50, db: Session = Depends(get_db)):
    return db.query(CheckHistory).order_by(CheckHistory.timestamp.desc()).limit(limit).all()
