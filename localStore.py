import os
import cv2
import uuid
import json
import faiss
import numpy as np
from glob import glob
from insightface.app import FaceAnalysis

# ---------------------------
# Step 1: Collect image paths
# ---------------------------
def collect_image_paths(dataset_dir):
    image_data = []
    for uid in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, uid)
        if not os.path.isdir(person_dir):
            continue
        for img_path in glob(os.path.join(person_dir, "*.jpg")):
            image_data.append({
                "uid": uid,
                "path": img_path,
                "image_name": os.path.basename(img_path)
            })
    return image_data

dataset_path = r"D:/Wrishav/face-recognition/datasets/AIML and DA/train"
image_data_list = collect_image_paths(dataset_path)

# -------------------------------------
# Step 2: Face embedding using InsightFace
# -------------------------------------
app = FaceAnalysis(name='buffalo_s')
app.prepare(ctx_id=0)  # Use -1 for CPU

def extract_embedding(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if len(faces) == 0:
        return None
    faces.sort(key=lambda x: x.det_score, reverse=True)
    return faces[0].embedding.astype("float32")

# -------------------------------------
# Step 3: Build FAISS Index (Cosine Similarity)
# -------------------------------------
dimension = 512  # ArcFace output
index = faiss.IndexFlatIP(dimension)  # Cosine similarity
embeddings = []
vector_ids = []
metadata_store = {}

def normalize(vec):
    return vec / np.linalg.norm(vec)

for item in image_data_list:
    embedding = extract_embedding(item["path"])
    if embedding is None:
        print(f"[ERROR] No face detected in {item['path']}")
        continue

    embedding = normalize(embedding)
    vector_id = str(uuid.uuid4())

    embeddings.append(embedding)
    vector_ids.append(vector_id)

    metadata_store[vector_id] = {
        "uid": item["uid"],
        "image_name": item["image_name"],
        "path": item["path"]
    }

# Convert to numpy array and add to FAISS index
embeddings_np = np.array(embeddings)
index.add(embeddings_np)

# -------------------------------------
# Step 4: Save FAISS index and metadata
# -------------------------------------
faiss.write_index(index, "face_index_cosine.faiss")

with open("face_metadata.json", "w") as f:
    json.dump(metadata_store, f, indent=4)

print(f"[INFO] Stored {len(embeddings)} face embeddings to FAISS index.")