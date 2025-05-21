
# Real-Time Face Recognition with InsightFace and Pinecone

A real-time facial recognition system built using [InsightFace](https://github.com/deepinsight/insightface) for face detection and feature extraction, and [Pinecone](https://www.pinecone.io/) as a vector similarity search engine to identify individuals from a known database.

This project captures live webcam frames, extracts 512-d face embeddings using a pre-trained ArcFace model, and performs real-time identity matching against a Pinecone vector index.

---

## 🔍 Features

- 📸 Real-time face detection and recognition via webcam
- 🤖 High-performance ArcFace model (`buffalo_l`) via InsightFace
- 🌐 Vector similarity search using Pinecone (cosine distance)
- 🎯 Accurate face matching with configurable thresholds
- 🚦 "Unknown" tagging for unrecognized faces
- 🖥️ GPU/CPU support (configurable)
- ✅ Easy to install and run with a single command

---

## 🗂️ Project Structure

```
face_recognition_realtime/
├── face_recognition_app/
│   ├── __init__.py
│   └── main.py              # ← Real-time inferencing logic
├── requirements.txt         # ← Dependencies
├── setup.py                 # ← Packaging and CLI entrypoint
└── README.md                # ← This file
```

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/face_recognition_realtime.git
cd face_recognition_realtime
```

### 2. Install the Package

> Ensure Python 3.7+ is installed in your environment.

```bash
pip install .
```

---

## 🚀 Running the Application

### Prerequisite
Before you run the app, ensure you:
- Have already uploaded face embeddings to Pinecone from your training dataset.
- Have your Pinecone API Key and Region.
- Replace the placeholder Pinecone API key in `face_recognition_app/main.py`.

### Run the real-time recognition app:

```bash
realtime-face-search
```

The app will:
- Open your default webcam
- Detect faces in real-time
- Match them against your Pinecone vector index
- Display UID and match score on screen

Press `q` to quit the stream.

---

## ⚙️ Configuration

Edit the `main()` function in `main.py` to customize parameters:

```python
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_REGION = "us-east-1"
INDEX_NAME = "aiml-da-face-embeds"
TOP_K = 1
SIMILARITY_THRESHOLD = 0.5
USE_GPU = 0  # Use -1 for CPU
```

---

## 📋 Example Output

When a face is detected and recognized, the display shows:

```
[INFO] Detected: uid123 (0.87)
```

If the similarity score is below the threshold:

```
[INFO] Detected: Unknown
```

---

## 🛠️ Dependencies

These are handled via `setup.py` but listed here for clarity:

- `insightface`
- `opencv-python`
- `pinecone-client`
- `numpy`
- `matplotlib`

Install manually if needed:

```bash
pip install -r requirements.txt
```

---

## 🧠 Notes

- Ensure your Pinecone index is **created and populated** before running real-time inference.
- This app only displays the **most similar** (`top_k=1`) match for each detected face.
- For multiple faces, the system processes all detected faces independently.
- You can use a GPU for faster processing if `USE_GPU=0` and your environment supports it.

---

## 🔒 Security

Avoid hardcoding secrets (e.g., API keys). You can use environment variables for production use:

```bash
export PINECONE_API_KEY="your-secret-key"
```

And update the code accordingly:

```python
import os
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
```

---

## 📈 Future Enhancements

- Face recognition for groups or crowds
- Logging and report generation
- Web dashboard using Streamlit or Flask
- Dockerization for platform-independent deployments

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙌 Acknowledgements

- [InsightFace](https://github.com/deepinsight/insightface) for facial feature extraction
- [Pinecone](https://www.pinecone.io/) for fast and scalable vector search

---

## 👤 Author

**Wrishav** – [GitHub](https://github.com/yourusername)

---
