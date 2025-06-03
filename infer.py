import os
import cv2
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from pinecone import Pinecone

query = input("[INPUT] Enter query image path: ")

QUERY_IMAGE_PATH = query
PINECONE_API_KEY = "pcsk_55iQ62_GvUCSHpRXAhy566mkXebjbFxSfe68aPWbZH2T93kboKuFLYy9tZwCotgAbbS8iM"
PINECONE_REGION = "us-east-1"
INDEX_NAME = "aiml-da-face-embeds"
TOP_K = 5  # Number of top results to return

print("[INFO] Loading face embedding model.")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)  # GPU: 0, CPU: -1
print("[INFO] Model loaded.")

def extract_embedding(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if not faces:
        raise ValueError("[ERROR] No face detected in query image.")
    return faces[0].embedding, img

try:
    query_embedding, query_img = extract_embedding(QUERY_IMAGE_PATH)
except ValueError as e:
    print(f"[ERROR] {e}")
    exit(1)
print(query_embedding.shape, query_img.shape)

print("[INFO] Connecting to Pinecone.")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

print("[INFO] Querying Pinecone for nearest neighbors.")
results = index.query(
    vector=query_embedding.tolist(),
    top_k=TOP_K,
    include_metadata=True
)

print("\n[OUTPUT] Top matches:")
if not results.matches:
    print("[ERROR ]No matches found.")
else:
    for match in results.matches:
        metadata = match.metadata
        print(f"[OUTPUT] UID: {metadata['uid']} | Image: {metadata['image_name']} | Score: {match.score:.4f}")