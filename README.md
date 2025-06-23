# Face Recognition Attendance System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)]()
[![Built with InsightFace](https://img.shields.io/badge/Built%20With-InsightFace-orange)](https://github.com/deepinsight/insightface)

## Overview

A face recognition-based attendance system using FAISS for fast similarity search and InsightFace for robust facial embeddings. It also integrates ElevenLabs TTS for personalized voice greetings and Pinecone for cloud embedding search (optional).

## Features

- Real-time face detection and recognition
- Local FAISS index for high-speed search
- Time-based check-in/out with CSV logging
- Text-to-speech greetings using ElevenLabs
- Pinecone-based remote query interface
- Cambria font overlay for professional UI display

## Setup

### Prerequisites

- Python 3.8+
- Webcam device
- CUDA-enabled GPU (optional for acceleration)

### Installation

```bash
git clone https://github.com/WrishavSett/Face-Recognition-Embeddings.git
cd Face-Recognition-Embeddings
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file with:

```ini
PINECONE_API_KEY=your_pinecone_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

## Usage

### Run Full Application with Webcam

```bash
python app.py
```

### Run in Headless Mode

```bash
python app.py --headless
```

### Query via Image

```bash
python test.py
```

## Directory Structure

```
.
├── app.py
├── test.py
├── track.py
├── utils.py
├── faissIndex/
│   ├── face_index_cosine.faiss
│   └── face_metadata.json
├── helper/
│   └── cambria.ttc
├── log/
│   └── attendance_log.csv
└── temp/
```

## Acknowledgements

- [InsightFace](https://github.com/deepinsight/insightface)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Pinecone](https://www.pinecone.io/)
- [ElevenLabs](https://www.elevenlabs.io/)

## License

This project is licensed under the MIT License.