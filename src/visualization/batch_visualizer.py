import cv2
import os
import sys
import numpy as np
import tensorflow as tf
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
MODEL_PATH     = BASE_DIR / "models" / "pose_model_best_v3.keras"
POSE_TASK_PATH = BASE_DIR / "models" / "pose_landmarker_lite.task"
RAW_DATA_DIR   = BASE_DIR / "data" / "raw"
OUTPUT_DIR     = BASE_DIR / "data" / "output_inference"
BUFFER_SIZE    = 40 

# --- DRAW SKELETON (Identik dengan live_inference) ---
def draw_skeleton(frame, results):
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

def process_video(video_path, output_path, model, landmarker):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    landmarks_buffer = []
    current_prediction = "Waiting..."
    subclass_name = ""
    confidence = 0.0
    color = (255, 255, 255)
    
    pbar = tqdm(total=total_frames, desc=f"Processing {video_path.name}", leave=False)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocessing frame untuk MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results   = landmarker.detect(mp_image)
        
        if results.pose_landmarks:
            frame_landmarks = []
            # Ambil 17 landmark pertama (0-16) seperti saat training
            for lm in results.pose_landmarks[0][:17]:
                frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
            
            landmarks_buffer.append(frame_landmarks)
            if len(landmarks_buffer) > BUFFER_SIZE:
                landmarks_buffer.pop(0)
            
            # Draw Skeleton
            draw_skeleton(frame, results)
            
            # Inference
            if len(landmarks_buffer) >= SEQUENCE_LENGTH:
                input_data = prepare_model_input(landmarks_buffer)
                prediction_probs = model.predict(input_data, verbose=0)[0]
                
                result = decode_prediction(prediction_probs)
                current_prediction = result["parent"]
                subclass_name      = result["subclass"]
                confidence         = result["confidence"]
                color              = (0, 255, 0) if result["is_fokus"] else (0, 0, 255)
        
        # Overlay UI
        cv2.putText(frame, f"Status: {current_prediction}", (30, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"Behavior: {subclass_name}", (30, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Conf: {confidence:.2f}", (30, 125), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        out.write(frame)
        pbar.update(1)
        
    cap.release()
    out.release()
    pbar.close()

def main():
    # 1. Load Model
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model tidak ditemukan di: {MODEL_PATH}")
        return
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("Keras model loaded.")

    # 2. Setup MediaPipe Landmarker
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

    # 3. Scan & Process Videos
    video_files = list(RAW_DATA_DIR.rglob("*.mp4"))
    print(f"Ditemukan {len(video_files)} video. Memilih 1 per subclass...")
    
    # Keyword list (identik dengan yang ada di filename)
    BEHAVIOR_KEYWORDS = [
        "menggunakan_ponsel",
        "melihat_layar",
        "membaca_materi",
        "menoleh",
        "menulis",
        "tidur",
    ]
    
    processed_subclasses = set()
    
    for vid_path in video_files:
        vid_name = vid_path.name.lower()
        
        # Tentukan subclass dari nama file
        current_subclass = None
        for kw in BEHAVIOR_KEYWORDS:
            if kw in vid_name:
                current_subclass = kw
                break
        
        if current_subclass is None:
            continue # Skip jika tidak ada keyword behavior
            
        # Jika subclass ini sudah diproses, lewati
        if current_subclass in processed_subclasses:
            continue
            
        rel_path = vid_path.relative_to(RAW_DATA_DIR)
        out_path = OUTPUT_DIR / rel_path.parent
        out_path.mkdir(parents=True, exist_ok=True)
        
        final_out_path = out_path / f"inference_{vid_path.name}"
        
        if final_out_path.exists():
            print(f"Skipping {vid_path.name} (sudah ada).")
            processed_subclasses.add(current_subclass) # Tandai sudah ada
            continue
            
        process_video(vid_path, final_out_path, model, landmarker)
        processed_subclasses.add(current_subclass) # Tandai sudah diproses
        
        # Jika semua subclass sudah ketemu, bisa berhenti (opsional, tapi biar aman lanjut scan)
        if len(processed_subclasses) == len(BEHAVIOR_KEYWORDS):
            print("Semua subclass sudah diproses satu video. Selesai.")
            break

if __name__ == "__main__":
    main()
