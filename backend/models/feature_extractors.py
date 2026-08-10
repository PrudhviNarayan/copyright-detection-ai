import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sentence_transformers import SentenceTransformer
import librosa
import numpy as np
import cv2
from PIL import Image

# Initialize Text Model
try:
    text_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading text model: {e}")
    text_model = None

# Initialize Image Model (ResNet50 feature extractor)
try:
    weights = models.ResNet50_Weights.DEFAULT
    image_model = models.resnet50(weights=weights)
    # Remove the classification layer to get feature embeddings (2048 dims for ResNet50)
    image_model = torch.nn.Sequential(*list(image_model.children())[:-1])
    image_model.eval()
    
    image_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
except Exception as e:
    print(f"Error loading image model: {e}")
    image_model = None

def get_text_embedding(text: str) -> np.ndarray:
    """Extracts embeddings for a given text."""
    if text_model is None:
        raise RuntimeError("Text model not initialized.")
    embedding = text_model.encode([text])[0]
    return embedding / np.linalg.norm(embedding)  # Normalize for cosine similarity

def get_image_embedding(image: Image.Image) -> np.ndarray:
    """Extracts embeddings for a given PIL Image."""
    if image_model is None:
        raise RuntimeError("Image model not initialized.")
    image_tensor = image_preprocess(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        embedding = image_model(image_tensor).numpy().flatten()
    return embedding / np.linalg.norm(embedding)

def get_audio_embedding(audio_path: str) -> np.ndarray:
    """Extracts embeddings for an audio file using MFCCs."""
    y, sr = librosa.load(audio_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    # Aggregate over time to get a fixed size vector (e.g., mean and std)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    embedding = np.concatenate([mfccs_mean, mfccs_std])
    if np.linalg.norm(embedding) == 0:
        return embedding
    return embedding / np.linalg.norm(embedding)

def get_video_embedding(video_path: str, max_frames=10) -> np.ndarray:
    """Extracts embeddings for a video by averaging frame embeddings."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, max(0, total_frames - 1), max_frames, dtype=int)
    
    embeddings = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        emb = get_image_embedding(pil_img)
        embeddings.append(emb)
    
    cap.release()
    if not embeddings:
        raise ValueError("Could not extract any frames from the video.")
    
    # Aggregate embeddings (mean pooling)
    video_embedding = np.mean(embeddings, axis=0)
    return video_embedding / np.linalg.norm(video_embedding)
