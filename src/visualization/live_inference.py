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
from collections import deque, Counter

# --- FIX FOR KERAS 3 COMPATIBILITY ---
from tensorflow.keras.layers import Dense, BatchNormalization

# Patch Dense
original_dense_init = Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    return original_dense_init(self, *args, **kwargs)
Dense.__init__ = patched_dense_init

# Patch BatchNormalization (Fix for 'renorm' error)
original_bn_init = BatchNormalization.__init__
def patched_bn_init(self, *args, **kwargs):
    kwargs.pop('renorm', None)
    kwargs.pop('renorm_clipping', None)
    kwargs.pop('renorm_momentum', None)
    return original_bn_init(self, *args, **kwargs)
BatchNormalization.__init__ = patched_bn_init

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

MODEL_PATH = BASE_DIR / "models" / "pose_model_holistic_local.keras"  # Model trained on 273 features
HOLISTIC_TASK_PATH = BASE_DIR / "models" / "holistic_landmarker.task"
BUFFER_SIZE    = 60   # Rolling buffer

# Import Holistic logic
from src.features.preprocess_holistic import extract_and_normalize_live, prepare_model_input_holistic
from src.features.holistic_config import TOTAL_FEATURES_PER_FRAME, SEQUENCE_LENGTH

# --- DRAW SKELETON (Simplified for Holistic) ---
def draw_landmarks(frame, results):
    if not results.pose_landmarks:
        return
    h, w, _ = frame.shape
    # Draw Pose
    for lm in results.pose_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
    # Draw Hands
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 2, (255, 0, 0), -1)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 2, (255, 0, 0), -1)

# --- MAIN ---
def main():
    print("=== Live Inference (Holistic) ===")
    print(f"Model     : {MODEL_PATH.name}")
    print(f"Features  : {TOTAL_FEATURES_PER_FRAME}")

    # 1. Load model
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model tidak ditemukan di: {MODEL_PATH}")
        return
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("Keras model loaded.")

    # 2. Setup MediaPipe Holistic
    if not HOLISTIC_TASK_PATH.exists():
        print(f"[ERROR] Holistic task tidak ditemukan di: {HOLISTIC_TASK_PATH}")
        return
    
    base_options = python.BaseOptions(model_asset_path=str(HOLISTIC_TASK_PATH))
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5
    )
    landmarker = vision.HolisticLandmarker.create_from_options(options)
    print("MediaPipe Holistic initialized.")

    # 3. Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Tidak bisa membuka webcam.")
        return

    landmarks_buffer   = []
    prediction_history = deque(maxlen=15)
    
    current_prediction = "Waiting..."
    subclass_name      = ""
    confidence         = 0.0
    color              = (255, 255, 255)
    no_detection_count = 0
    prev_pose          = None # Untuk fallback tangan

    print("--- LIVE INFERENCE STARTED --- Tekan 'q' untuk keluar.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame     = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect Holistic
        results = landmarker.detect(mp_image)

        if results.pose_landmarks:
            no_detection_count = 0
            
            # Extract & Normalize (273 features)
            flat_features, current_pose = extract_and_normalize_live(results, prev_pose)
            prev_pose = current_pose
            
            landmarks_buffer.append(flat_features)
            if len(landmarks_buffer) > BUFFER_SIZE:
                landmarks_buffer.pop(0)

            draw_landmarks(frame, results)

            # Inference
            if len(landmarks_buffer) >= SEQUENCE_LENGTH:
                input_data = prepare_model_input_holistic(landmarks_buffer)
                prediction_probs = model.predict(input_data, verbose=0)[0]
                
                # Decode
                result = decode_prediction(prediction_probs)
                prediction_history.append(result["class_idx"])
                
                most_common_idx = Counter(prediction_history).most_common(1)[0][0]
                
                from src.visualization.inference_utils import (
                    SUBCLASS_NAMES, SUBCLASS_TO_PARENT, FOKUS_CLASSES
                )
                
                subclass_name      = SUBCLASS_NAMES[most_common_idx]
                current_prediction = SUBCLASS_TO_PARENT[most_common_idx].upper()
                is_fokus           = most_common_idx in FOKUS_CLASSES
                confidence         = result["confidence"] if result["class_idx"] == most_common_idx else 0.0
                
                color = (0, 255, 0) if is_fokus else (0, 0, 255)
        else:
            no_detection_count += 1
            # Reset buffer jika tidak terdeteksi >30 frame berturut-turut
            if no_detection_count > 30:
                landmarks_buffer.clear()
                current_prediction = "No Pose"
                subclass_name      = ""
                confidence         = 0.0
                color              = (200, 200, 200)

        # --- UI Overlay ---
        # Background hitam semi-transparan untuk teks
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 5), (400, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.putText(frame, f"Status: {current_prediction}",
                    (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"Behavior: {subclass_name}",
                    (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}",
                    (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(frame, f"Buffer: {len(landmarks_buffer)}/{BUFFER_SIZE}",
                    (30, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("Focus Estimation — Live Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Inference selesai.")


if __name__ == "__main__":
    main()
