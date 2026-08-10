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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For frontend access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure temp directory exists
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

@app.on_event("startup")
def startup_event():
    init_db()
    generate_all() 

class TextCheckRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "Backend is running 🚀"}



def save_upload_file(upload_file: UploadFile) -> str:
    ext = os.path.splitext(upload_file.filename)[1]
    filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(TEMP_DIR, filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return filepath

def determine_risk(score: float, threshold: float) -> str:
    if score >= threshold:
        return "HIGH"
    return "LOW"

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

@app.post("/check-text")
def check_text(request: TextCheckRequest, threshold: float = 0.8, db: Session = Depends(get_db)):
    try:
        embedding = get_text_embedding(request.text)
        indexer = get_indexer("text")
        
        results = indexer.search(embedding, k=1)
        score = 0.0
        matched_with = "None"
        
        if results:
            meta, score = results[0]
            matched_with = meta["filename"]
            
        risk = determine_risk(score, threshold)
        
        # We use a snippet of text as the filename for logging
        snippet = request.text[:30] + "..." if len(request.text) > 30 else request.text
        log_history(db, snippet, "text", score, risk, matched_with)
        
        return {
            "similarity_score": score,
            "top_match": matched_with,
            "risk_level": risk
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check-image")
def check_image(file: UploadFile = File(...), threshold: float = Form(0.8), db: Session = Depends(get_db)):
    try:
        filepath = save_upload_file(file)
        pil_img = Image.open(filepath)
        embedding = get_image_embedding(pil_img)
        
        indexer = get_indexer("image")
        results = indexer.search(embedding, k=1)
        
        score = 0.0
        matched_with = "None"
        if results:
            meta, score = results[0]
            matched_with = meta["filename"]
            
        risk = determine_risk(score, threshold)
        log_history(db, file.filename, "image", score, risk, matched_with)
        
        # Clean up
        os.remove(filepath)
        
        return {
            "similarity_score": score,
            "top_match": matched_with,
            "risk_level": risk
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check-audio")
def check_audio(file: UploadFile = File(...), threshold: float = Form(0.8), db: Session = Depends(get_db)):
    try:
        filepath = save_upload_file(file)
        embedding = get_audio_embedding(filepath)
        
        indexer = get_indexer("audio")
        results = indexer.search(embedding, k=1)
        
        score = 0.0
        matched_with = "None"
        if results:
            meta, score = results[0]
            matched_with = meta["filename"]
            
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

@app.post("/check-video")
def check_video(file: UploadFile = File(...), threshold: float = Form(0.8), db: Session = Depends(get_db)):
    try:
        filepath = save_upload_file(file)
        embedding = get_video_embedding(filepath)
        
        indexer = get_indexer("video")
        results = indexer.search(embedding, k=1)
        
        score = 0.0
        matched_with = "None"
        if results:
            meta, score = results[0]
            matched_with = meta["filename"]
            
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

@app.get("/history")
def get_history(limit: int = 50, db: Session = Depends(get_db)):
    history = db.query(CheckHistory).order_by(CheckHistory.timestamp.desc()).limit(limit).all()
    return history
