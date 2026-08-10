import os
import sys
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from backend.models.feature_extractors import (
    get_text_embedding, get_image_embedding, get_audio_embedding, get_video_embedding
)
from backend.faiss_index.indexer import get_indexer

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
AUG_DIR = os.path.join(DATASET_DIR, "augmented")

def calculate_metrics(y_true, y_pred, y_score, threshold=0.7):
    # True positives: prediction matches ground truth AND score >= threshold
    # False positives: prediction does NOT match ground truth OR score < threshold (when it should match)
    
    tp = sum(1 for t, p, s in zip(y_true, y_pred, y_score) if t == p and s >= threshold)
    fp = sum(1 for t, p, s in zip(y_true, y_pred, y_score) if (t != p and s >= threshold) or (t == p and s < threshold))
    fn = sum(1 for t, p, s in zip(y_true, y_pred, y_score) if s < threshold) # In a pure retrieval context, missed matches.
    
    # Simplified metrics for demonstration
    total = len(y_true)
    accuracy = tp / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1

def evaluate_modality(modality, get_embedding_fn, ext):
    print(f"\n--- Evaluating {modality.upper()} ---")
    indexer = get_indexer(modality)
    
    y_true = []
    y_pred = []
    y_score = []
    
    if not os.path.exists(AUG_DIR):
        print("No augmented dataset found.")
        return
        
    files = [f for f in os.listdir(AUG_DIR) if f.startswith("aug_") and modality in f and f.endswith(ext)]
    
    for f in files:
        filepath = os.path.join(AUG_DIR, f)
        
        try:
            if modality == "text":
                with open(filepath, "r") as file:
                    emb = get_embedding_fn(file.read())
            elif modality == "image":
                emb = get_embedding_fn(Image.open(filepath))
            else:
                emb = get_embedding_fn(filepath)
                
            results = indexer.search(emb, k=1)
            
            ground_truth = f.replace("aug_", "")
            y_true.append(ground_truth)
            
            if results:
                meta, score = results[0]
                y_pred.append(meta["filename"])
                y_score.append(score)
            else:
                y_pred.append("None")
                y_score.append(0.0)
                
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    if y_true:
        acc, prec, rec, f1 = calculate_metrics(y_true, y_pred, y_score)
        print(f"Accuracy:  {acc:.2f}")
        print(f"Precision: {prec:.2f}")
        print(f"Recall:    {rec:.2f}")
        print(f"F1 Score:  {f1:.2f}")
    else:
        print("No files to evaluate.")

if __name__ == "__main__":
    evaluate_modality("text", get_text_embedding, ".txt")
    evaluate_modality("image", get_image_embedding, ".jpg")
    evaluate_modality("audio", get_audio_embedding, ".wav")
    evaluate_modality("video", get_video_embedding, ".avi")
