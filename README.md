# 🚀 Multi-Modal Copyright Detection System

> **Resume Headline:** An end-to-end full-stack AI system for multi-modal copyright detection, leveraging deep learning embeddings and FAISS for scalable vector similarity search across text, images, audio, and video.

Welcome to the **Multi-Modal Copyright Detection System**. This is a complete, production-ready full-stack AI system designed to identify copyright violations across a variety of media formats: **Text, Image, Audio, and Video**. 

By utilizing deep learning embeddings and **FAISS (Facebook AI Similarity Search)**, this system performs ultra-fast, high-accuracy semantic comparisons to detect infringement, even when the original material has been modified.

---

## 📖 Overview

Traditional copyright detection relies on exact matching or basic hashing, which fails when content is altered. This system uses **similarity-based detection**:
- **Paraphrased Text:** Detects similar meaning, not just exact words.
- **Modified Images:** Identifies cropped, color-shifted, or compressed images.
- **Edited Media:** Detects trimmed, noisy, or slightly altered audio and video clips.

This approach provides a robust defense against intellectual property theft in the modern digital landscape.

---

## ✨ Features

- **Multi-Modal Detection:** Support for checking Text, Images, Audio, and Video files.
- **FAISS Similarity Search:** Lightning-fast vector similarity search across large datasets.
- **Embedding-Based Comparison:** Powered by state-of-the-art models (SentenceTransformers, ResNet50, MFCCs).
- **Similarity Graph:** Dynamic visual representation of match confidence using Chart.js.
- **Threshold Slider:** Adjustable strictness for matching (0.0 to 1.0).
- **Risk Classification:** Automatic grading into High, Medium, or Low risk based on the similarity score.
- **Upload History:** Persistent tracking of previous checks powered by SQLite.
- **JSON Report Download:** Easily export detection reports.
- **Dark UI:** A sleek, modern, glassmorphism-inspired dark interface.

---

## 🧠 Architecture

```text
User Upload (Text/Image/Audio/Video)
        ↓
Feature Extraction
        ↓
Embedding Vector
        ↓
FAISS Index Search
        ↓
Top Similar Matches
        ↓
Similarity Score (%)
        ↓
Risk Classification
        ↓
Frontend Visualization + History Storage
```

---

## 🛠️ Tech Stack

**Backend:**
- **FastAPI:** High-performance async web framework.
- **FAISS:** Vector database for similarity search.
- **SentenceTransformers & Torch:** Deep learning text and image embeddings.
- **Librosa:** Audio feature extraction.
- **OpenCV:** Video frame processing.
- **SQLite (SQLAlchemy):** Relational database for history tracking.

**Frontend:**
- **HTML, CSS, JS:** Vanilla web stack for maximum performance.
- **Chart.js:** Real-time data visualization.

---

## 📂 Project Structure

```text
copyright-detection-ai/
├── backend/
│   ├── faiss_index/         # FAISS indexing logic and saved indices
│   ├── models/              # Feature extraction models (ResNet50, MiniLM, etc.)
│   ├── database.py          # SQLite setup and SQLAlchemy models
│   └── main.py              # FastAPI application and endpoints
├── dataset/
│   ├── raw/                 # Original generated media
│   ├── augmented/           # Modified media for testing
│   └── generate_dataset.py  # Script to synthesize and index dummy data
├── frontend/
│   ├── index.html           # Main UI template
│   ├── style.css            # Dark mode styling
│   └── script.js            # Frontend logic and API integration
├── requirements.txt         # Python dependencies
├── evaluate.py              # Script to calculate Accuracy, Precision, Recall, F1
└── README.md                # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone Repo
```bash
git clone https://github.com/yourusername/copyright-detection-ai.git
cd copyright-detection-ai
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```
*(Note: Audio and Video processing may require system-level installations of `ffmpeg`)*

### 4. Run Dataset Generator
Generate synthetic data and build the initial FAISS indices:
```bash
python dataset/generate_dataset.py
```

### 5. Start Backend
Launch the FastAPI server:
```bash
uvicorn backend.main:app --reload
```
The API and Swagger docs will be available at `http://localhost:8000/docs`.

### 6. Run Frontend
Serve the UI locally:
```bash
cd frontend
python -m http.server 8080
```
Open your browser and navigate to `http://localhost:8080`.

---

## 🚀 Deployment Instructions (Render)

This project is configured for easy deployment on [Render](https://render.com) using Docker, ensuring that system dependencies like `ffmpeg` (for audio/video processing) are correctly installed.

### Step-by-Step Guide
1. **Push to GitHub**: Ensure all your code, including `Dockerfile` and `render.yaml`, is pushed to your GitHub repository.
2. **Connect to Render**: Log in to Render and create a new **Blueprint Instance**.
3. **Connect Repository**: Select your GitHub repository containing this project.
4. **Deploy**: Render will automatically detect the `render.yaml` file, build the Docker container (which installs Python dependencies and `ffmpeg`), and start the FastAPI backend on port `10000`.
5. **Update Frontend**: Once the backend is live, copy the Render URL (e.g., `https://your-backend-url.onrender.com`), update the `API_BASE` variable in `frontend/script.js`, and host your frontend anywhere (e.g., GitHub Pages, Netlify, or Vercel).

---

## 📊 Evaluation

The system includes a benchmarking script to measure model performance against the augmented dataset.
Run the evaluation suite:
```bash
python evaluate.py
```
This generates a report containing:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

---

## 📸 Screenshots

### User Interface
![UI Overview](/images/ui.png)

### Similarity Graph
![Similarity Graph](/images/graph.png)

### Results Panel
![Results Panel](/images/result.png)

---

## 🔮 Future Improvements

- **Cloud Deployment:** Containerize and deploy to AWS/GCP with managed databases.
- **Model Upgrades:** Implement **CLIP** for cross-modal search and **Whisper** for advanced audio transcription matching.
- **Docker Support:** Create a `docker-compose.yml` for true one-click startup.
- **Distributed FAISS:** Scale out the vector database for millions of records.
