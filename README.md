
# Face Recognition Embeddings

A modular, end-to-end system for facial recognition using facial embeddings, vector search with Pinecone, and real-time/video inference. This project leverages deep learning-based face embeddings (via InsightFace) to identify individuals based on facial similarity, with additional features like voice-based greetings.

---

## 🧱 Project Structure

```plaintext
face-recognition/
├── .gitignore
├── README.md
├── requirements.txt
├── train.py                # Embedding extraction & indexing in Pinecone
├── infer.py                # Image-based face matching
├── setup.py
├── directory_structure.txt
├── app/
│   ├── app.py             # App-style interface for greeting and name lookup
│   ├── main.py            # Real-time webcam-based face recognition
│   ├── test.py            # Video file processing with recognition
│   └── utils.py           # Utilities: UID-name mapping, greeting audio
├── datasets/
│   └── AIML and DA/
│       ├── train/         # Training images (folders per person)
│       ├── test/          # Test set (same UID folders)
│       └── validate/      # Validation set (same UID folders)
├── helper/
│   ├── cambria.ttc        # Font for video overlays
│   └── test.mp4           # Sample input video
└── temp/
    └── *.mp3              # Output voice greetings
```

---

## 📦 Requirements

Install all Python dependencies via:

```bash
pip install -r requirements.txt
```

### Required Dependencies:

- `opencv-python`
- `insightface`
- `pinecone-client`
- `numpy`
- `matplotlib`
- `onnxruntime` for CPU or `onnxruntime-gpu` for GPU

> ℹ️ Make sure to have access to a **Pinecone account** and **ElevenLabs API key** (for voice greetings). These keys should be stored securely (e.g., using a `.env` file).

---

## 🚀 How It Works

### 1. **Face Embedding Generation & Indexing**
Run `train.py` to:
- Load all training images from the `datasets/AIML and DA/train` directory.
- Detect faces using `insightface`.
- Extract 512-dimensional embeddings.
- Upload them to **Pinecone**, a vector database, along with metadata (UID, image name).

```bash
python train.py
```

---

### 2. **Query via Image**
Run `infer.py` to:
- Load an input image.
- Extract face embedding.
- Query Pinecone for top similar faces.
- Display UID, image name, and similarity score.

```bash
python infer.py
```

---

### 3. **Application Workflow**
The `app/` folder contains modular scripts for enhanced use cases:

#### ✅ `app.py`
- Accepts a query image path.
- Retrieves the most similar UID from Pinecone.
- Maps UID → Name using a hardcoded dictionary.
- Greets the user with a personalized audio message using **ElevenLabs TTS**.

#### ✅ `main.py`
- Activates your webcam.
- Detects and recognizes faces in real-time.
- Displays bounding boxes and UID labels.

#### ✅ `test.py`
- Processes a video file (`test.mp4`).
- Annotates recognized faces and FPS.
- Saves output to `output.mp4`.

---

### 4. **Name & Audio Utilities**
The `utils.py` file provides:
- UID to name mapping (`get_name`)
- Time-aware greetings
- Text-to-speech greeting generator (`generate_voice`)

---

## 🔐 API Keys & Security

This project uses:
- **Pinecone API key** for embedding search
- **ElevenLabs API key** for voice synthesis

> ⚠️ Keys are hardcoded in the codebase. **Please replace them with environment variables and load via `.env` for production or public use.**

---

## ✨ Future Improvements

- Add a REST API using FastAPI or Flask.
- Store UID-name mappings in a database.
- Add a web interface for registration and querying.
- Dockerize for deployment.

---

## 📬 Contact

Developed by [Wrishav Sett](https://github.com/WrishavSett)
