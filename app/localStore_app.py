import os
import cv2
import time
import json
import numpy as np
from dotenv import load_dotenv
from PIL import ImageFont, ImageDraw, Image
from insightface.app import FaceAnalysis
from utils import get_name, get_current_time, play_sound
from track import add_to_dictionary
import faiss

load_dotenv()

# --- Configuration ---
INDEX_PATH = "./faissIndex/face_index_cosine.faiss"
METADATA_PATH = "./faissIndex/face_metadata.json"
TOP_K = 1
SIMILARITY_THRESHOLD = 0.5  # Cosine similarity threshold
USE_GPU = 0  # Use -1 for CPU

welcome_dictionary = {}
goodbye_dictionary = {}

CAMBRIA_FONT_PATH = "./helper/cambria.ttc"
font_size = 24
if not os.path.exists(CAMBRIA_FONT_PATH):
    raise FileNotFoundError(f"[ERROR] Cambria font file not found: {CAMBRIA_FONT_PATH}")
font = ImageFont.truetype(CAMBRIA_FONT_PATH, font_size)

# --- FAISS and Metadata Initialization ---
if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
    raise FileNotFoundError("[ERROR] FAISS index or metadata file not found.")

print("[INFO] Loading FAISS index and metadata.")
faiss_index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)
vector_ids = list(metadata.keys())
print("[INFO] FAISS and metadata loaded.")

# --- Face Model Initialization ---
print("[INFO] Loading face embedding model.")
facemodel = FaceAnalysis(name='buffalo_l')
facemodel.prepare(ctx_id=USE_GPU)
print("[INFO] Model loaded.")

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("[ERROR] Cannot open webcam. Make sure it's connected and not in use.")
print("[INFO] Webcam feed started. Press 'q' to quit.")

# --- Helper Function ---
def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

# --- Main Loop ---
while True:
    start_time = time.time()
    date_str, time_str = get_current_time()

    ret, frame = cap.read()
    if not ret:
        print("[INFO] Failed to read frame from webcam.")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = facemodel.get(img_rgb)

    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = normalize(face.embedding.astype("float32")).reshape(1, -1)

        try:
            scores, indices = faiss_index.search(embedding, TOP_K)
            best_score = float(scores[0][0])
            best_idx = int(indices[0][0])

            if best_score >= SIMILARITY_THRESHOLD:
                vector_id = vector_ids[best_idx]
                meta = metadata.get(vector_id, {})
                uid = meta.get("uid", "Unknown")
                label = f"{uid} ({best_score:.2f})"

                # Greeting Logic
                if "08:45:00" <= time_str < "17:45:00":
                    exists, welcome_dictionary = add_to_dictionary(welcome_dictionary, uid)
                    if not exists:
                        print(f"[INFO] New UID detected: {uid}. Adding to welcome dictionary.")
                        name = get_name(uid)
                        play_sound(uid)
                        print(f"[INFO] Welcome message generated for {name} with UID {uid}.")
                        goodbye_dictionary = {}  # Reset goodbye
                elif "17:45:00" <= time_str < "23:59:59":
                    exists, goodbye_dictionary = add_to_dictionary(goodbye_dictionary, uid)
                    if not exists:
                        print(f"[INFO] New UID detected: {uid}. Adding to goodbye dictionary.")
                        name = get_name(uid)
                        play_sound(uid)
                        print(f"[INFO] Goodbye message generated for {name} with UID {uid}.")
                        welcome_dictionary = {}  # Reset welcome
            else:
                label = "Unknown"
        except Exception as e:
            label = f"Error: {str(e)}"
            print(f"[ERROR] FAISS query failed: {e}")

        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)

        # Draw with PIL
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((bbox[0], bbox[1] - font_size - 5), label, font=font, fill=color)
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Draw rectangle with OpenCV
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    # Show live frame
    cv2.imshow('Webcam Face Recognition (Local FAISS)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Welcome entries: {welcome_dictionary}")
print(f"[INFO] Goodbye entries: {goodbye_dictionary}")
print("[INFO] Processing complete. Exiting.")