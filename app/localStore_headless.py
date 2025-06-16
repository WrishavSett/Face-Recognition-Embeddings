import cv2
import os
import time
import json
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from insightface.app import FaceAnalysis
import faiss

def load_faiss_index(index_path, metadata_path):
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError("FAISS index or metadata file not found.")
    
    print("[INFO] Loading FAISS index and metadata.")
    index = faiss.read_index(index_path)
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    print("[INFO] FAISS index and metadata loaded.")
    return index, metadata

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def main():
    INDEX_PATH = "./face_index_cosine.faiss"
    METADATA_PATH = "./face_metadata.json"
    VIDEO_PATH = "./helper/test.mp4"
    OUTPUT_VIDEO_PATH = "./helper/localSore_output.mp4"
    CAMBRIA_FONT_PATH = "C:/Windows/Fonts/cambria.ttc"  # Change path for non-Windows

    TOP_K = 1
    SIMILARITY_THRESHOLD = 0.5  # cosine similarity
    USE_GPU = 0  # -1 for CPU

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"[ERROR] Video file not found: {VIDEO_PATH}")
    if not os.path.exists(CAMBRIA_FONT_PATH):
        raise FileNotFoundError(f"[ERROR] Font file not found: {CAMBRIA_FONT_PATH}")

    font = ImageFont.truetype(CAMBRIA_FONT_PATH, 24)

    # Load model
    print("[INFO] Loading InsightFace model.")
    app = FaceAnalysis(name='buffalo_s')
    app.prepare(ctx_id=USE_GPU)
    print("[INFO] Model loaded.")

    # Load FAISS + metadata
    index, metadata = load_faiss_index(INDEX_PATH, METADATA_PATH)
    vector_ids = list(metadata.keys())

    # Setup video reader/writer
    cap = cv2.VideoCapture(VIDEO_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    print(f"[INFO] Processing video. Saving to {OUTPUT_VIDEO_PATH}")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video.")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)

        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = normalize(face.embedding.astype("float32")).reshape(1, -1)

            scores, indices = index.search(embedding, TOP_K)
            best_score = float(scores[0][0])
            best_idx = int(indices[0][0])

            if best_score >= SIMILARITY_THRESHOLD:
                vector_id = vector_ids[best_idx]
                meta = metadata.get(vector_id, {})
                uid = meta.get("uid", "Unknown")
                label = f"{uid} ({best_score:.2f})"
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            # Draw bounding box and label
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Add FPS overlay
        end_time = time.time()
        fps_val = 1.0 / (end_time - start_time + 1e-5)
        fps_text = f"FPS: {fps_val:.2f}"

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((10, 10), fps_text, font=font, fill=(255, 255, 0))
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        out.write(frame)

    cap.release()
    out.release()
    print(f"[INFO] Processing complete. Output saved to {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()