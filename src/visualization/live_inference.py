import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys
import os
from pathlib import Path
import time

# --- FIX FOR KERAS 3 COMPATIBILITY ---
from tensorflow.keras.layers import Dense
original_dense_init = Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    return original_dense_init(self, *args, **kwargs)
Dense.__init__ = patched_dense_init

# --- PATH SETUP ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

# Import shared utils (normalisasi identik dengan training pipeline)
from src.visualization.inference_utils import (
    prepare_model_input,
    SEQUENCE_LENGTH,
    TOTAL_FEATURES,
)

# --- KONFIGURASI ---
MODEL_PATH     = BASE_DIR / "models" / "pose_model_best.keras"
POSE_TASK_PATH = BASE_DIR / "models" / "pose_landmarker_lite.task"
BUFFER_SIZE    = 40   # Rolling buffer lebih besar dari SEQUENCE_LENGTH


# --- DRAW SKELETON ---
def draw_landmarks(frame, results):
    if not results.pose_landmarks:
        return
    connections = [
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    h, w, _ = frame.shape
    for landmarks in results.pose_landmarks:
        for idx in range(17):
            lm = landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        for s, e in connections:
            if s < 17 and e < 17:
                p1, p2 = landmarks[s], landmarks[e]
                cv2.line(frame,
                         (int(p1.x * w), int(p1.y * h)),
                         (int(p2.x * w), int(p2.y * h)),
                         (0, 255, 0), 2)


# --- MAIN ---
def main():
    print("=== Live Inference (Webcam) ===")
    print(f"Model     : {MODEL_PATH.name}")
    print(f"Seq Len   : {SEQUENCE_LENGTH}, Features: {TOTAL_FEATURES}")

    # 1. Load model
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model tidak ditemukan di: {MODEL_PATH}")
        return
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("Keras model loaded.")

    # 2. Setup MediaPipe
    if not POSE_TASK_PATH.exists():
        print(f"[ERROR] Pose task tidak ditemukan di: {POSE_TASK_PATH}")
        return
    base_options = python.BaseOptions(model_asset_path=str(POSE_TASK_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    print("MediaPipe Landmarker initialized.")

    # 3. Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Tidak bisa membuka webcam.")
        return

    landmarks_buffer   = []
    current_prediction = "Waiting..."
    confidence         = 0.0
    color              = (255, 255, 255)
    no_detection_count = 0

    print("--- LIVE INFERENCE STARTED --- Tekan 'q' untuk keluar.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror flip (same as training data capture orientation)
        frame     = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results   = landmarker.detect(mp_image)

        if results.pose_landmarks:
            no_detection_count = 0
            frame_landmarks = []
            for lm in results.pose_landmarks[0][:17]:
                frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

            landmarks_buffer.append(frame_landmarks)
            if len(landmarks_buffer) > BUFFER_SIZE:
                landmarks_buffer.pop(0)

            draw_landmarks(frame, results)

            # Inference
            if len(landmarks_buffer) >= SEQUENCE_LENGTH:
                input_data = prepare_model_input(landmarks_buffer)
                prediction = model.predict(input_data, verbose=0)[0][0]

                confidence         = prediction if prediction > 0.5 else 1.0 - prediction
                current_prediction = "FOKUS" if prediction > 0.5 else "TIDAK FOKUS"
                color              = (0, 255, 0) if prediction > 0.5 else (0, 0, 255)
        else:
            no_detection_count += 1
            # Reset buffer jika tidak terdeteksi >30 frame berturut-turut
            if no_detection_count > 30:
                landmarks_buffer.clear()
                current_prediction = "No Pose"
                confidence         = 0.0
                color              = (200, 200, 200)

        # --- UI Overlay ---
        # Background hitam semi-transparan untuk teks
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (350, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.putText(frame, f"Status: {current_prediction}",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}",
                    (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Buffer: {len(landmarks_buffer)}/{BUFFER_SIZE}",
                    (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("Focus Estimation — Live Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Inference selesai.")


if __name__ == "__main__":
    main()
