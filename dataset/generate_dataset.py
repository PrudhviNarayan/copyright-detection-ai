import os
import sys
import numpy as np
import cv2
import scipy.io.wavfile as wavfile
from PIL import Image, ImageEnhance, ImageFilter

# Add backend to path to import extractors and indexer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.models.feature_extractors import get_text_embedding, get_image_embedding, get_audio_embedding, get_video_embedding
from backend.faiss_index.indexer import get_indexer

DATASET_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(DATASET_DIR, "raw")
AUG_DIR = os.path.join(DATASET_DIR, "augmented")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(AUG_DIR, exist_ok=True)

# 1. TEXT DATASET
raw_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the future of technology.",
    "Deep learning models require large amounts of high-quality data."
]

def generate_text():
    print("Generating Text Data...")
    indexer = get_indexer("text")
    for i, t in enumerate(raw_texts):
        filename = f"text_{i}.txt"
        with open(os.path.join(RAW_DIR, filename), "w") as f:
            f.write(t)
        emb = get_text_embedding(t)
        indexer.add_item(emb, filename)
        
        # Augmented
        aug_text = t.replace("lazy", "sleepy").replace("technology", "tech industry")
        with open(os.path.join(AUG_DIR, f"aug_{filename}"), "w") as f:
            f.write(aug_text)

# 2. IMAGE DATASET
def generate_images():
    print("Generating Image Data...")
    indexer = get_indexer("image")
    for i in range(3):
        # Create random color image
        color = tuple(np.random.randint(0, 255, 3).tolist())
        img = Image.new("RGB", (224, 224), color)
        filename = f"image_{i}.jpg"
        img.save(os.path.join(RAW_DIR, filename))
        
        emb = get_image_embedding(img)
        indexer.add_item(emb, filename)
        
        # Augmented: blur and brightness
        aug_img = img.filter(ImageFilter.GaussianBlur(2))
        enhancer = ImageEnhance.Brightness(aug_img)
        aug_img = enhancer.enhance(0.8)
        aug_img.save(os.path.join(AUG_DIR, f"aug_{filename}"))

# 3. AUDIO DATASET
def generate_audio():
    print("Generating Audio Data...")
    indexer = get_indexer("audio")
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), False)
    
    for i in range(3):
        # Generate sine wave
        freq = 440.0 + (i * 100)
        audio = np.sin(freq * 2 * np.pi * t)
        audio = (audio * 32767).astype(np.int16)
        
        filename = f"audio_{i}.wav"
        raw_path = os.path.join(RAW_DIR, filename)
        wavfile.write(raw_path, sr, audio)
        
        emb = get_audio_embedding(raw_path)
        if np.linalg.norm(emb) > 0:
            indexer.add_item(emb, filename)
        
        # Augmented: add noise
        noise = np.random.normal(0, 5000, audio.shape)
        aug_audio = np.clip(audio + noise, -32768, 32767).astype(np.int16)
        wavfile.write(os.path.join(AUG_DIR, f"aug_{filename}"), sr, aug_audio)

# 4. VIDEO DATASET
def generate_video():
    print("Generating Video Data...")
    indexer = get_indexer("video")
    for i in range(2):
        filename = f"video_{i}.avi"
        raw_path = os.path.join(RAW_DIR, filename)
        out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (224, 224))
        
        color = (0, 255, 0) if i == 0 else (0, 0, 255)
        for _ in range(30):
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frame[:] = color
            out.write(frame)
        out.release()
        
        emb = get_video_embedding(raw_path)
        indexer.add_item(emb, filename)
        
        # Augmented: add noise to frames
        aug_path = os.path.join(AUG_DIR, f"aug_{filename}")
        out_aug = cv2.VideoWriter(aug_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (224, 224))
        for _ in range(30):
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frame[:] = color
            noise = np.random.randint(0, 50, (224,224,3), dtype=np.uint8)
            frame = cv2.add(frame, noise)
            out_aug.write(frame)
        out_aug.release()

if __name__ == "__main__":
    print("Starting dataset generation and indexing...")
    generate_text()
    generate_images()
    generate_audio()
    generate_video()
    print("Dataset generated and FAISS indices updated.")
