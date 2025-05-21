import os
from glob import glob

def collect_image_paths(dataset_dir="dataset"):
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

image_data_list = collect_image_paths("/content/drive/MyDrive/Datasets/AIML and DA/train")

import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l')  # Loads ArcFace model
app.prepare(ctx_id=0)  # 0 = GPU; use -1 for CPU

def extract_embedding(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if len(faces) == 0:
        return None
    return faces[0].embedding  # 512-d float vector

import os
from pinecone import Pinecone, ServerlessSpec

# Replace with your actual key and region
api_key = "pcsk_55iQ62_GvUCSHpRXAhy566mkXebjbFxSfe68aPWbZH2T93kboKuFLYy9tZwCotgAbbS8iM"
region = "us-east-1"

pc = Pinecone(api_key=api_key)

# Create index if it doesn't exist
index_name = "aiml-da-face-embeds"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=512,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",         # or "gcp"
            region=region        # must match your Pinecone project region
        )
    )

# Connect to the index
index = pc.Index(index_name)

import uuid

to_upsert = []

for item in image_data_list:
    embedding = extract_embedding(item["path"])
    if embedding is None:
        print(f"[WARNING] No face detected in {item['path']}")
        continue

    vector_id = str(uuid.uuid4())

    metadata = {
        "uid": item["uid"],
        "image_name": item["image_name"],
        "path": item["path"]
    }

    to_upsert.append((vector_id, embedding.tolist(), metadata))

# Batch upsert
BATCH_SIZE = 100
print(f"[INFO] Uploading {len(to_upsert)} embeddings to Pinecone...")

for i in range(0, len(to_upsert), BATCH_SIZE):
    index.upsert(vectors=to_upsert[i:i + BATCH_SIZE])

print("[INFO] All embeddings uploaded successfully.")