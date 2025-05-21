import cv2
import os
from insightface.app import FaceAnalysis
from pinecone import Pinecone
import numpy as np

def main():
    PINECONE_API_KEY = "pcsk_55iQ62_GvUCSHpRXAhy566mkXebjbFxSfe68aPWbZH2T93kboKuFLYy9tZwCotgAbbS8iM"
    PINECONE_REGION = "us-east-1"
    INDEX_NAME = "aiml-da-face-embeds"
    TOP_K = 1
    USE_GPU = 0  # Use -1 for CPU
    SIMILARITY_THRESHOLD = 0.5

    print("[INFO] Loading face embedding model.")
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=USE_GPU)
    print("[INFO] Model loaded.")

    print("[INFO] Connecting to Pinecone.")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    print("[INFO] Connected to Pinecone.")

    cap = cv2.VideoCapture(0)  # 0 for default webcam
    print("[INFO] Starting video stream. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)

        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding.tolist()

            # Query Pinecone
            try:
                results = index.query(vector=embedding, top_k=TOP_K, include_metadata=True)
                if results.matches:
                    top_match = results.matches[0]
                    score = top_match.score
                    if score >= SIMILARITY_THRESHOLD:
                        uid = top_match.metadata['uid']
                        label = f"{uid} ({score:.2f})"
                    else:
                        label = f"Unknown"
                else:
                    label = "Unknown"
            except Exception as e:
                label = f"Error: {str(e)}"

            # Draw bounding box and label
            color = (0, 255, 0) if label != "Below Threshold" else (0, 0, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Real-Time Face Search", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting.")
            break

    cap.release()
    cv2.destroyAllWindows()