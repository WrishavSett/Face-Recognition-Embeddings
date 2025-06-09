import os
import cv2
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

def query_face_embedding():
    query = input("[INPUT] Enter query image path: ")

    QUERY_IMAGE_PATH = query
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_REGION = "us-east-1"
    INDEX_NAME = "aiml-da-face-embeds"
    TOP_K = 5  # Number of top results to return
    USE_GPU = 0  # Use -1 for CPU
    SIMILARITY_THRESHOLD = 0.5

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

    try:
        print("[INFO] Querying Pinecone for nearest neighbors.")
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=TOP_K,
            include_metadata=True
        )

        print("\n[OUTPUT] Top matches:")
        if not results.matches:
            print("[ERROR] No matches found.")
            return 0
        else:
            for match in results.matches:
                # metadata = match.metadata
                # print(f"- UID: {metadata['uid']} | Image: {metadata['image_name']} | Score: {match.score:.4f}")
                top_match = match
                score = top_match.score
                if score >= SIMILARITY_THRESHOLD:
                    uid = top_match.metadata['uid']
                    print(f"[OUTPUT] UID: {uid} | Score: {score:.4f}")
                    return uid
                else:
                    print(f"[ERROR] No matches found.")
                    return 0
    except Exception as e:
        print(f"[ERROR] An error occurred: {str(e)}")
        return 0
    
def get_name():
    uid = query_face_embedding()
    from utils import get_name
    if uid:
        name = get_name(uid)
        print(f"[OUTPUT] Name: {name}")
        return name
    else:
        print("[ERROR] No valid UID found.")
        return None

def greet_user():
    from utils import generate_voice
    uid = query_face_embedding()
    if uid:
        generate_voice(uid)
    else:
        print("[ERROR] No valid UID found. Cannot generate greeting voice.")

greet_user()