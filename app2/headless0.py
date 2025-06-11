import cv2
import os
import time
from PIL import ImageFont, ImageDraw, Image
from insightface.app import FaceAnalysis
from pinecone import Pinecone
import faiss
import json
from dotenv import load_dotenv
import numpy as np

load_dotenv()

def main():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_REGION = "us-east-1"
    INDEX_NAME = "aiml-da-face-embeds"
    TOP_K = 1
    USE_GPU = 0  # Use -1 for CPU
    SIMILARITY_THRESHOLD = 10
    REFRESH_INDEX = False # Set to True to refresh the FAISS index

    INDEX_PATH = "./app2/data/index.faiss"  # Path to your FAISS index file
    UID_MAP_PATH = "./app2/data/uid_map.json"  # Path to your UID map file
    VIDEO_PATH = "./app2/data/test.mp4"  # Path to your input video
    OUTPUT_VIDEO_PATH = "./app2/data/output.mp4"
    CAMBRIA_FONT_PATH = "C:/Windows/Fonts/cambria.ttc"  # Adjust for your OS

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"[ERROR] Video file not found: {VIDEO_PATH}")
    if not os.path.exists(CAMBRIA_FONT_PATH):
        raise FileNotFoundError(f"[ERROR] Cambria font file not found: {CAMBRIA_FONT_PATH}")
    font = ImageFont.truetype(CAMBRIA_FONT_PATH, size=24)

    print("[INFO] Loading face embedding model.")
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=USE_GPU)
    print("[INFO] Model loaded.")

    if os.path.exists(INDEX_PATH) and not REFRESH_INDEX:
        index = faiss.read_index(INDEX_PATH)
        with open(UID_MAP_PATH) as f:
            uid_map = json.load(f)
    else:
        print("[INFO] Connecting to Pinecone.")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pc_index = pc.Index(INDEX_NAME)
        print("[INFO] Connected to Pinecone.")

        print("[INFO] Downloading all vectors from Pinecone index...")
        all_ids = [id for page in pc_index.list() for id in page]
        vectors = []; uid_map = {}
        for i, id in enumerate(all_ids): # Adjust the slice to limit the number of vectors for testing
            response = pc_index.fetch(ids=[id]).vectors[id] 
            vectors.append(response.values)
            uid_map[i] = response.metadata["uid"]
        print(f"[INFO] Downloaded {len(vectors)} vectors.")

        index = faiss.IndexFlatIP(512)
        vectors = np.array(vectors)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        index.add(vectors)

        faiss.write_index(index, INDEX_PATH)
        with open(UID_MAP_PATH, "w") as f:
            json.dump(uid_map, f)
    print("[INFO] FAISS index loaded")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"[ERROR] Cannot open video: {VIDEO_PATH}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    print(f"[INFO] Processing and saving to: {OUTPUT_VIDEO_PATH}")

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video or failed to read frame.")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)

        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = np.array(face.embedding).reshape(1, -1)

            try:
                similarities, ids = index.search(embedding, TOP_K)
                if ids[0][0] != -1:
                    score = similarities[0][0]
                    if score >= SIMILARITY_THRESHOLD:
                        uid = uid_map[str(ids[0][0])]
                        label = f"{uid} ({score:.2f})"
                    else:
                        label = "Unknown"
                else:
                    label = "Unknown"
            except Exception as e:
                label = f"Error: {str(e)}"

            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Calculate and overlay FPS using Cambria font
        end_time = time.time()
        elapsed_time = end_time - start_time
        current_fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0
        fps_text = f"FPS: {current_fps:.2f}"

        # Use PIL to overlay custom font
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((10, 10), fps_text, font=font, fill=(255, 255, 0))
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Save to output video
        out.write(frame)

    cap.release()
    out.release()
    print(f"[INFO] Video saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()