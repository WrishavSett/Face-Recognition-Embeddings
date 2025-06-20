import os
import cv2
import time
import json
import csv
import argparse
import numpy as np
import platform
import sys
from dotenv import load_dotenv
from PIL import ImageFont, ImageDraw, Image
from insightface.app import FaceAnalysis
from utils import get_name, get_current_time, play_sound
from track import add_to_dictionary
import faiss

# --- Platform-specific keypress handler ---
if platform.system() == 'Windows':
    import msvcrt
    def check_keypress():
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key in [b'q', b'\x1b']:  # 'q' or Esc
                return True
        return False
else:
    import select
    import termios
    import tty
    def check_keypress():
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            key = sys.stdin.read(1)
            if key in ['q', '\x1b']:  # 'q' or Esc
                return True
        return False

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Face Recognition Attendance System")
parser.add_argument("--headless", action="store_true", help="Run without displaying the webcam feed")
parser.add_argument("--faissgpu", action="store_true", help="Enable FAISS GPU acceleration if available")
args = parser.parse_args()

# Terminal setup for Unix-like systems
if not platform.system() == 'Windows' and args.headless:
    orig_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)

load_dotenv()

# --- Configuration ---
INDEX_PATH = "./faissIndex/face_index_cosine.faiss"
METADATA_PATH = "./faissIndex/face_metadata.json"
CAMBRIA_FONT_PATH = "./helper/cambria.ttc"
LOG_FILE = "./log/attendance_log.csv"

TOP_K = 1
SIMILARITY_THRESHOLD = 0.5
USE_GPU = 0  # InsightFace: use -1 for CPU, 0 for GPU

welcome_dictionary = {}
goodbye_dictionary = {}
day_end_logged = False

# --- Font Setup ---
if not os.path.exists(CAMBRIA_FONT_PATH):
    raise FileNotFoundError(f"[ERROR] Cambria font file not found: {CAMBRIA_FONT_PATH}")
font = ImageFont.truetype(CAMBRIA_FONT_PATH, 24)

# --- Load FAISS + Metadata ---
if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
    raise FileNotFoundError("[ERROR] FAISS index or metadata file not found.")

print("[INFO] Loading FAISS index and metadata.")
faiss_index = faiss.read_index(INDEX_PATH)

if args.faissgpu:
    try:
        res = faiss.StandardGpuResources()
        faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
        print("[INFO] FAISS GPU enabled.")
    except Exception as e:
        print(f"[WARNING] FAISS GPU not available or failed to initialize. Falling back to CPU. Error: {e}")

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
print("[INFO] Webcam feed started. Press 'q' or 'Esc' to quit.")

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
        logout_time = entry['last_seen']
        for goodbye_entry in goodbye_dictionary.values():
            if goodbye_entry['uid'] == uid:
                logout_time = goodbye_entry['last_seen']
                break
        write_log(uid, name, log_in, logout_time)
        logged_uids.add(uid)
    print(f"[INFO] Logged {len(logged_uids)} users for logout.")

# --- Main Loop ---
try:
    while True:
        date_str, time_str = get_current_time()

        if time_str == "06:00:00":
            print("[INFO] Resetting tracking dictionaries at 06:00.")
            welcome_dictionary.clear()
            goodbye_dictionary.clear()
            day_end_logged = False

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
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            draw.text((bbox[0], bbox[1] - 30), label, font=font, fill=color)
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        if not args.headless:
            cv2.imshow("Webcam Face Recognition (Local)", frame)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                print("[INFO] Exit key pressed. Terminating...")
                break
        else:
            if check_keypress():
                print("[INFO] Exit key pressed (headless mode). Terminating.")
                break

finally:
    if not platform.system() == 'Windows' and args.headless:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Welcome entries: {welcome_dictionary}")
    print(f"[INFO] Goodbye entries: {goodbye_dictionary}")
    print("[INFO] Exiting application.")
