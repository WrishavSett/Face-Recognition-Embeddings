import cv2
import os
import time
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from insightface.app import FaceAnalysis
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

def main():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_REGION = "us-east-1"
    INDEX_NAME = "aiml-da-face-embeds"
    TOP_K = 1
    USE_GPU = 0  # Use -1 for CPU
    SIMILARITY_THRESHOLD = 0.5

    VIDEO_PATH = "./helper/test.mp4"  # Path to your input video
    OUTPUT_VIDEO_PATH = "./helper/output.mp4"
    CAMBRIA_FONT_PATH = "C:/Windows/Fonts/cambria.ttc"  # Adjust for your OS

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"[ERROR] Video file not found: {VIDEO_PATH}")
    if not os.path.exists(CAMBRIA_FONT_PATH):
        raise FileNotFoundError(f"[ERROR] Cambria font file not found: {CAMBRIA_FONT_PATH}")

    font_size = 24
    font = ImageFont.truetype(CAMBRIA_FONT_PATH, font_size)

    print("[INFO] Loading face embedding model.")
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=USE_GPU)
    print("[INFO] Model loaded.")

    print("[INFO] Connecting to Pinecone.")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    print("[INFO] Connected to Pinecone.")

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
            embedding = face.embedding.tolist()

            try:
                results = index.query(vector=embedding, top_k=TOP_K, include_metadata=True)
                if results.matches:
                    top_match = results.matches[0]
                    score = top_match.score
                    if score >= SIMILARITY_THRESHOLD:
                        uid = top_match.metadata.get('uid', 'Unknown')
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