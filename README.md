# Face Recognition with Pinecone & Audio Greetings

A modular, real-time face recognition system using deep-learning embeddings (via InsightFace), Pinecone vector search, and personalized voice greetings with ElevenLabs. This project also includes attendance-like logging and time-aware greeting behavior.

---

## Project Structure

```plaintext
face-recognition/
├── README.md
├── requirements.txt
├── train.py                # Embedding extraction & indexing in Pinecone
├── infer.py                # Image-based face query
├── app/
│   ├── app.py             # Full video-based recognition + greeting logic
│   ├── headless.py        # Lightweight video processing (no audio)
│   ├── main.py            # Real-time webcam recognition
│   ├── test.py            # CLI: image-based UID+greeting
│   ├── track.py           # UID detection and logging dictionary
│   └── utils.py           # UID-name mapping, greeting generation, TTS
├── datasets/
│   └── AIML and DA/
│       ├── train/
│       ├── test/
│       └── validate/
├── helper/
│   ├── cambria.ttc        # Font used in video overlays
│   └── test.mp4           # Sample input video
└── temp/
    └── *.mp3              # Generated audio greetings
```

---

## Requirements & Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Recommended Python: `>=3.8`

### Key Libraries

- `opencv-python`
- `insightface`
- `pinecone-client`
- `elevenlabs`
- `onnxruntime` or `onnxruntime-gpu` or `onnxruntime-silicon`
- `numpy`, `PIL`, `matplotlib`, `playsound`, `soundfile`

---

## API Keys & Environment Variables

**Create a `.env` file** at the root:

```bash
PINECONE_API_KEY=your_pinecone_key
ELEVENLABS_API_KEY=your_elevenlabs_key
```

Update your code to load them:

```python
from dotenv import load_dotenv
load_dotenv()
import os
api_key = os.getenv("PINECONE_API_KEY")
```

---

## Core Functionality

### 🔧 1. Train Embeddings

```bash
python train.py
```

- Loads images from `datasets/AIML and DA/train/`
- Detects faces and generates 512-D embeddings
- Indexes embeddings in Pinecone with metadata (UID, filename)

---

### 2. Query via Image (CLI)

```bash
python app/test.py
```

- Prompts for a local image path
- Displays top matching UID and name
- Greets the user with generated voice (via ElevenLabs)

---

## Internals: How Greeting & Tracking Works

- `track.py` maintains two dictionaries: `welcome_dictionary` and `goodbye_dictionary`
- Each new UID is only greeted once per session
- Time-based logic (e.g., "Welcome" before 17:45, "Goodbye" after)
- `generate_voice()` uses ElevenLabs API to generate mp3 greetings
- `get_name()` maps UIDs to full names using a hardcoded dictionary in `utils.py`

---

## Developer Notes

- Models used: `insightface.buffalo_l` for embedding
- Vector DB: [Pinecone](https://www.pinecone.io/)
- Text-to-speech: [ElevenLabs](https://www.elevenlabs.io/)
- Experimental TTS: Kokoro

---

## Example Workflow

```bash
# Step 1: Train and index embeddings
python train.py

# Step 2: Test with a query image
python app/test.py

# Step 3: Run real-time recognition
python app/main.py

OR

# Step 3: Process recorded video with complete time-based greeting logic
python app/app.py
```

---

## Contact

Developed by [Wrishav Sett](https://github.com/WrishavSett)

> ⚠️ For production, **remove all hardcoded API keys** and use `.env` + `dotenv`.

---
