# Multi-Modal Copyright Detection System

A full-stack, production-style web application for detecting copyright violations across text, images, audio, and video using modern deep learning embeddings and FAISS vector similarity search.

## Features

- **Multi-Modal Support**: Detects similarities in Text, Images, Audio, and Video.
- **Deep Learning Embeddings**: 
  - Text: `SentenceTransformers` (`all-MiniLM-L6-v2`)
  - Image: `ResNet50` (via PyTorch/Torchvision)
  - Audio: `librosa` (MFCC features)
  - Video: Frame extraction via OpenCV + ResNet50 mean pooling
- **Fast Similarity Search**: Powered by FAISS (Facebook AI Similarity Search).
- **History Tracking**: SQLite database via SQLAlchemy to store history and scores.
- **Modern UI**: Dark theme, glassmorphism UI, tabbed navigation, threshold slider, and similarity visualization using Chart.js.

## Project Structure

```
copyright-detection-ai/
├── backend/
│   ├── faiss_index/         # FAISS indexing logic and saved indices
│   ├── models/              # Feature extraction models
│   ├── database.py          # SQLite setup and models
│   ├── main.py              # FastAPI endpoints
├── dataset/
│   ├── generate_dataset.py  # Synthetic data generation and augmentation
│   ├── raw/                 # Generated base dataset
│   ├── augmented/           # Augmented variants for evaluation
├── frontend/
│   ├── index.html           # UI
│   ├── style.css            # Dark theme styling
│   ├── script.js            # API interaction and Chart.js logic
├── requirements.txt         # Python dependencies
├── evaluate.py              # Script to calculate Accuracy, Precision, Recall, F1 Score
└── README.md
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd copyright-detection-ai
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   *Note: Video and Audio processing requires system-level libraries. Make sure you have `ffmpeg` installed if `opencv` or `librosa` fail to read media files.*

## Generating the Dataset

Before running the backend, you should generate the synthetic dataset and populate the FAISS indices.

```bash
python dataset/generate_dataset.py
```

This will create `text`, `image`, `audio`, and `video` indices inside `backend/faiss_index/indices/`.

## Running the Application

### 1. Start the Backend (FastAPI)

```bash
uvicorn backend.main:app --reload
```
The API will run at `http://localhost:8000`. You can view the interactive Swagger API documentation at `http://localhost:8000/docs`.

### 2. Start the Frontend

You can run a simple HTTP server to serve the frontend files:

```bash
cd frontend
python -m http.server 8080
```
Open your browser and navigate to `http://localhost:8080`.

## Evaluation

To evaluate the models using the generated augmented dataset, run:

```bash
python evaluate.py
```
This will print Accuracy, Precision, Recall, and F1 Score for all modalities.

## API Endpoints

- `POST /check-text`: Accepts JSON `{ "text": "..." }` and `threshold` query param.
- `POST /check-image`: Accepts `multipart/form-data` with `file` and `threshold`.
- `POST /check-audio`: Accepts `multipart/form-data` with `file` and `threshold`.
- `POST /check-video`: Accepts `multipart/form-data` with `file` and `threshold`.
- `GET /history`: Returns a JSON list of recent checks.

## Author

Generated for AI/ML Portfolio.
