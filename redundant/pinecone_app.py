import os
import cv2
import time
import numpy as np
from pinecone import Pinecone
from dotenv import load_dotenv
from track import add_to_dictionary # Assuming 'track.py' is in the same directory
from insightface.app import FaceAnalysis
from PIL import ImageFont, ImageDraw, Image
from utils import get_name, get_current_time, play_sound # Assuming 'utils.py' is in the same directory

load_dotenv()

# --- Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = "us-east-1"
INDEX_NAME = "aiml-da-face-embeds"
TOP_K = 1
USE_GPU = 0  # Use -1 for CPU
SIMILARITY_THRESHOLD = 0.5

# Dictionaries to track greetings
welcome_dictionary = {}
goodbye_dictionary = {}

# Font for displaying text (ensure it's accessible)
CAMBRIA_FONT_PATH = "./helper/cambria.ttc"  # Adjust for your OS

if not os.path.exists(CAMBRIA_FONT_PATH):
    raise FileNotFoundError(f"[ERROR] Cambria font file not found: {CAMBRIA_FONT_PATH}")

font_size = 24
font = ImageFont.truetype(CAMBRIA_FONT_PATH, font_size)

# --- Model and Pinecone Initialization ---
print("[INFO] Loading face embedding model.")
facemodel = FaceAnalysis(name='buffalo_l')
facemodel.prepare(ctx_id=USE_GPU)
print("[INFO] Model loaded.")

print("[INFO] Connecting to Pinecone.")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
print("[INFO] Connected to Pinecone.")

# --- Webcam Setup ---
# Use 0 for default webcam, or change to a different number if you have multiple webcams
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("[ERROR] Cannot open webcam. Make sure it's connected and not in use.")

print("[INFO] Starting webcam feed. Press 'q' to quit.")

# --- Real-time Processing Loop ---
while True:
    start_time = time.time()
    d, t = get_current_time()

    ret, frame = cap.read()
    if not ret:
        print("[INFO] Failed to read frame from webcam.")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = facemodel.get(img_rgb)

    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.embedding.tolist()

        try:
            results = index.query(vector=embedding, top_k=TOP_K, include_metadata=True)
            if results.matches:
                top_match = results.matches[0]
                score = top_match.score
                if score >= SIMILARITY_THRESHOLD:
                    uid = top_match.metadata.get('uid', 'Unknown')
                    label = f"{uid} ({score:.2f})"
                    if t >= "08:45:00" and t < "17:45:00":
                        exists, welcome_dictionary = add_to_dictionary(welcome_dictionary, uid)
                        if exists is False:
                            print(f"[INFO] New UID detected: {uid}. Adding to welcome dictionary.")
                            name = get_name(uid)
                            play_sound(uid)
                            print(f"[INFO] Welcome message generated for {name} with UID {uid}.")
                            goodbye_dictionary = {} # Reset goodbye if person just arrived
                    elif t >= "17:45:00" and t < "23:59:59":
                        exists, goodbye_dictionary = add_to_dictionary(goodbye_dictionary, uid)
                        if exists is False:
                            print(f"[INFO] New UID detected: {uid}. Adding to goodbye dictionary.")
                            name = get_name(uid)
                            play_sound(uid)
                            print(f"[INFO] Goodbye message generated for {name} with UID {uid}.")
                            welcome_dictionary = {} # Reset welcome if person just left
                else:
                    label = "Unknown"
            else:
                label = "Unknown"
        except Exception as e:
            label = f"Error: {str(e)}"
            print(f"[ERROR] Pinecone query failed: {e}")

        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Draw label using PIL for custom font
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((bbox[0], bbox[1] - font_size - 5), label, font=font, fill=color) # Adjusted position
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # Display the frame
    cv2.imshow('Webcam Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Welcome entries: {welcome_dictionary}")
print(f"[INFO] Goodbye entries: {goodbye_dictionary}")
print("[INFO] Processing complete. Exiting.")