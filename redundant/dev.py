import os
import cv2
import time
import json
import csv
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
CAMBRIA_FONT_PATH = "./helper/cambria.ttc"
LOG_FILE = "./log/attendance_log.csv"

TOP_K = 1
SIMILARITY_THRESHOLD = 0.5
USE_GPU = 0  # Use -1 for CPU

# Tracking dictionaries
welcome_dictionary = {}
goodbye_dictionary = {}
day_end_logged = False  # Reset daily

# --- Font Setup ---
if not os.path.exists(CAMBRIA_FONT_PATH):
    raise FileNotFoundError(f"[ERROR] Cambria font file not found: {CAMBRIA_FONT_PATH}")
font = ImageFont.truetype(CAMBRIA_FONT_PATH, 24)

# --- Load FAISS + Metadata ---
if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
    raise FileNotFoundError("[ERROR] FAISS index or metadata file not found.")

print("[INFO] Loading FAISS index and metadata.")
faiss_index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)
vector_ids = list(metadata.keys())
print("[INFO] FAISS and metadata loaded.")

# --- Load Face Detection Model ---
print("[INFO] Loading face embedding model.")
facemodel = FaceAnalysis(name='buffalo_l')
facemodel.prepare(ctx_id=USE_GPU)
print("[INFO] Model loaded.")

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("[ERROR] Cannot open webcam.")
print("[INFO] Webcam feed started. Press 'q' to quit.")

# --- Utility Functions ---
def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def write_log(uid, name, log_in, log_out):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['UID', 'Name', 'Login Time', 'Logout Time'])
        writer.writerow([uid, name, log_in, log_out])
    print(f"[INFO] Logged {uid}: {log_in} - {log_out}")

def check_and_log_day_end():
    print("[INFO] Running end-of-day logging.")
    logged_uids = set()

    for track_id, entry in welcome_dictionary.items():
        uid = entry['uid']
        name = entry['name']
        log_in = entry['time']

        # Check if user is in goodbye dictionary
        logout_time = entry['last_seen']
        for goodbye_entry in goodbye_dictionary.values():
            if goodbye_entry['uid'] == uid:
                logout_time = goodbye_entry['last_seen']
                break

        write_log(uid, name, log_in, logout_time)
        logged_uids.add(uid)

    print(f"[INFO] Logged {len(logged_uids)} users for logout.")

# --- Main Loop ---
while True:
    start_time = time.time()
    date_str, time_str = get_current_time()

    # Reset at 06:00
    if time_str == "06:00:00":
        print("[INFO] Resetting tracking dictionaries at 06:00.")
        welcome_dictionary.clear()
        goodbye_dictionary.clear()
        day_end_logged = False

    # End-of-day log at 01:00:00
    if time_str == "01:00:00" and not day_end_logged:
        check_and_log_day_end()
        day_end_logged = True

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

                # Greeting logic
                if "08:45:00" <= time_str < "17:45:00":
                    exists, welcome_dictionary = add_to_dictionary(welcome_dictionary, uid)
                    if not exists:
                        name = get_name(uid)
                        play_sound(uid)
                        print(f"[INFO] Welcome recorded for {name}")
                        goodbye_dictionary.clear()
                elif "17:45:00" <= time_str < "23:59:59":
                    exists, goodbye_dictionary = add_to_dictionary(goodbye_dictionary, uid)
                    if not exists:
                        name = get_name(uid)
                        play_sound(uid)
                        print(f"[INFO] Goodbye recorded for {name}")
                        welcome_dictionary.clear()
            else:
                label = "Unknown"

        except Exception as e:
            label = f"Error: {str(e)}"
            print(f"[ERROR] FAISS query failed: {e}")

        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)

        # Draw label with PIL
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((bbox[0], bbox[1] - 30), label, font=font, fill=color)
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    # Show video
    cv2.imshow("Webcam Face Recognition (Local)", frame)

    # Exit if 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Final Cleanup ---
cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Welcome entries: {welcome_dictionary}")
print(f"[INFO] Goodbye entries: {goodbye_dictionary}")
print("[INFO] Exiting application.")