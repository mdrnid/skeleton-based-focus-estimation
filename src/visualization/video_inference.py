import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys
import os
from pathlib import Path
from tqdm import tqdm
import argparse

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
    decode_prediction,
    SEQUENCE_LENGTH,
    TOTAL_FEATURES,
)

# --- KONFIGURASI ---
# --- KONFIGURASI DEFAULT ---
# Prefer v5 (robust) model, fall back to v4 if not yet trained
_MODEL_V5 = BASE_DIR / "models" / "pose_model_best_v5.keras"
_MODEL_V4 = BASE_DIR / "models" / "pose_model_best_v4_zip.keras"
DEFAULT_MODEL_PATH = _MODEL_V5 if _MODEL_V5.exists() else _MODEL_V4
POSE_TASK_PATH = BASE_DIR / "models" / "pose_landmarker_lite.task"
BUFFER_SIZE    = 40   # Lebih besar dari SEQUENCE_LENGTH agar resampling smooth


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
                p1 = landmarks[s]
                p2 = landmarks[e]
                cv2.line(frame,
                         (int(p1.x * w), int(p1.y * h)),
                         (int(p2.x * w), int(p2.y * h)),
                         (0, 255, 0), 2)


# --- MAIN ---
def process_video(input_path, output_path, model_path, no_flip=False):
    print(f"=== Video Inference ===")
    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")
    print(f"Model  : {model_path.name}")
    print(f"Seq Len: {SEQUENCE_LENGTH}, Features: {TOTAL_FEATURES}")

    # 1. Load model
    if not model_path.exists():
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        return
    model = tf.keras.models.load_model(str(model_path))
    print("Keras model loaded.")

    # 2. Setup MediaPipe
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

    # 3. Video I/O
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Tidak bisa membuka video: {input_path}")
        return

    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    landmarks_buffer   = []
    current_prediction = "Waiting..."
    subclass_name      = ""
    confidence         = 0.0
    color              = (255, 255, 255)

    pbar = tqdm(total=total_frames, desc="Processing Frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror flip (aktif secara default, nonaktifkan dengan --no-flip)
        if not no_flip:
            frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results   = landmarker.detect(mp_image)

        if results.pose_landmarks:
            frame_landmarks = []
            for lm in results.pose_landmarks[0][:17]:
                frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

            landmarks_buffer.append(frame_landmarks)
            if len(landmarks_buffer) > BUFFER_SIZE:
                landmarks_buffer.pop(0)

            draw_landmarks(frame, results)

            # Inference (mulai ketika minimal SEQUENCE_LENGTH frame terkumpul)
            if len(landmarks_buffer) >= SEQUENCE_LENGTH:
                input_data = prepare_model_input(landmarks_buffer)
                prediction_probs = model.predict(input_data, verbose=0)[0]
                
                # Decode multi-class prediction
                result = decode_prediction(prediction_probs)
                
                current_prediction = result["parent"]
                subclass_name      = result["subclass"]
                confidence         = result["confidence"]
                color              = (0, 255, 0) if result["is_fokus"] else (0, 0, 255)

        # --- Overlay ---
        cv2.putText(frame, f"Predict: {current_prediction}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        cv2.putText(frame, f"Behavior: {subclass_name}",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Conf: {confidence:.2f}",
                    (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Buffer: {len(landmarks_buffer)}/{BUFFER_SIZE}",
                    (30, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        out.write(frame)
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    print(f"\n✅ Selesai. Hasil disimpan di: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Focus Estimation — Video Inference")
    parser.add_argument("--input",   type=str, default="inference.mp4",
                        help="Path ke video input")
    parser.add_argument("--output",  type=str, default=None,
                        help="Path untuk menyimpan video output")
    parser.add_argument("--model",   type=str, default=None,
                        help="Path ke file model .keras")
    parser.add_argument("--no-flip", action="store_true",
                        help="Nonaktifkan mirror flip (aktif by default)")
    args = parser.parse_args()

    input_video  = args.input
    output_video = args.output if args.output else "scratch/inference_latest.mp4"
    
    # Pilih model
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = DEFAULT_MODEL_PATH

    process_video(input_video, output_video, model_path, no_flip=args.no_flip)
