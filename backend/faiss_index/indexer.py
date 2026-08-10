import faiss
import numpy as np
import pickle
import os

INDEX_DIR = os.path.join(os.path.dirname(__file__), "indices")
os.makedirs(INDEX_DIR, exist_ok=True)

class ModalityIndexer:
    def __init__(self, name: str, dim: int):
        self.name = name
        self.dim = dim
        self.index_path = os.path.join(INDEX_DIR, f"{name}.index")
        self.metadata_path = os.path.join(INDEX_DIR, f"{name}_meta.pkl")
        
        self.metadata = []  # List of dicts: {"id": int, "filename": str, "type": str}
        self.index = None
        self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"Loaded {self.name} index with {self.index.ntotal} items.")
        else:
            # Using Inner Product since we are normalizing embeddings (cosine similarity)
            self.index = faiss.IndexFlatIP(self.dim)
            print(f"Created new {self.name} index.")

    def add_item(self, embedding: np.ndarray, filename: str):
        assert embedding.shape == (self.dim,), f"Expected shape ({self.dim},), got {embedding.shape}"
        
        # Add to FAISS index
        self.index.add(np.expand_dims(embedding, axis=0).astype(np.float32))
        
        # Add metadata
        item_id = len(self.metadata)
        self.metadata.append({"id": item_id, "filename": filename, "type": self.name})
        
        self._save()

    def search(self, query_embedding: np.ndarray, k: int = 1):
        if self.index.ntotal == 0:
            return [], []
            
        assert query_embedding.shape == (self.dim,)
        
        # Search returns distances and indices
        distances, indices = self.index.search(np.expand_dims(query_embedding, axis=0).astype(np.float32), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                # Cosine distance is returned as inner product because vectors are normalized
                score = float(distances[0][i])
                # Ensure score is bound between 0 and 1
                score = max(0.0, min(1.0, score))
                results.append((self.metadata[idx], score))
                
        return results

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

# Dimensions based on models
TEXT_DIM = 384     # all-MiniLM-L6-v2
IMAGE_DIM = 2048   # ResNet50
AUDIO_DIM = 40     # 20 MFCC mean + 20 MFCC std
VIDEO_DIM = 2048   # ResNet50 (mean over frames)

# Singleton instances
text_indexer = ModalityIndexer("text", TEXT_DIM)
image_indexer = ModalityIndexer("image", IMAGE_DIM)
audio_indexer = ModalityIndexer("audio", AUDIO_DIM)
video_indexer = ModalityIndexer("video", VIDEO_DIM)

def get_indexer(modality: str) -> ModalityIndexer:
    if modality == "text":
        return text_indexer
    elif modality == "image":
        return image_indexer
    elif modality == "audio":
        return audio_indexer
    elif modality == "video":
        return video_indexer
    else:
        raise ValueError(f"Unknown modality: {modality}")
