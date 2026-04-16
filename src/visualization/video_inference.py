import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from pathlib import Path
from tqdm import tqdm
import argparse

# --- FIX FOR KERAS 3 COMPATIBILITY ERROR ---
from tensorflow.keras.layers import Dense
original_dense_init = Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    return original_dense_init(self, *args, **kwargs)
Dense.__init__ = patched_dense_init

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "pose_model_best_97acc.keras"
POSE_TASK_PATH = BASE_DIR / "models" / "pose_landmarker_lite.task"
SEQUENCE_LENGTH = 10
BUFFER_SIZE = 20

# --- PREPROCESSING UTILS ---
def resample_sequence(frames_features, target_length):
    n_frames, n_features = frames_features.shape
    if n_frames == target_length:
        return frames_features
    src_idx = np.arange(n_frames)
    target_idx = np.linspace(0, n_frames - 1, target_length)
    resampled = np.zeros((target_length, n_features))
    for i in range(n_features):
        resampled[:, i] = np.interp(target_idx, src_idx, frames_features[:, i])
    return resampled

def normalize_landmarks(landmarks_buffer):
    frames_features = []
    for landmarks in landmarks_buffer:
        coords = np.array(landmarks, dtype=np.float32).reshape(17, 4)
        nose = coords[0, 0:3]
        coords[:, 0:3] -= nose
        l_shoulder = coords[11, 0:3]
        r_shoulder = coords[12, 0:3]
        shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)
        if shoulder_width > 1e-6:
            coords[:, 0:3] /= shoulder_width
        frames_features.append(coords.flatten())
    return np.array(frames_features)

def draw_landmarks(frame, results):
    if not results.pose_landmarks:
        return
    connections = [
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 12)
    ]
    h, w, _ = frame.shape
    for landmarks in results.pose_landmarks:
        for idx in range(17):
            landmark = landmarks[idx]
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        for start_idx, end_idx in connections:
            if start_idx < 17 and end_idx < 17:
                p1 = landmarks[start_idx]
                p2 = landmarks[end_idx]
                pt1 = (int(p1.x * w), int(p1.y * h))
                pt2 = (int(p2.x * w), int(p2.y * h))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

# --- MAIN INFERENCE ---
def process_video(input_path, output_path):
    print(f"Initializing video inference: {input_path}")
    
    # 1. Load Keras Model
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("Keras model loaded.")

    # 2. Setup MediaPipe Landmarker
    base_options = python.BaseOptions(model_asset_path=str(POSE_TASK_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    print("MediaPipe Landmarker initialized.")

    # 3. Setup Video Capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 4. Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"Saving output to: {output_path}")

    landmarks_buffer = []
    current_prediction = "Waiting..."
    confidence = 0.0
    color = (255, 255, 255)

    pbar = tqdm(total=total_frames, desc="Processing Frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        results = landmarker.detect(mp_image)

        if results.pose_landmarks:
            frame_landmarks = []
            for landmark in results.pose_landmarks[0][:17]:
                frame_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            
            landmarks_buffer.append(frame_landmarks)
            if len(landmarks_buffer) > BUFFER_SIZE:
                landmarks_buffer.pop(0)

            draw_landmarks(frame, results)

            # Perform Inference
            if len(landmarks_buffer) >= 5:
                features_normalized = normalize_landmarks(landmarks_buffer)
                features_resampled = resample_sequence(features_normalized, SEQUENCE_LENGTH)
                input_data = features_resampled.reshape(1, SEQUENCE_LENGTH, 68)
                prediction = model.predict(input_data, verbose=0)[0][0]
                
                confidence = prediction if prediction > 0.5 else 1.0 - prediction
                current_prediction = "FOCUS" if prediction > 0.5 else "TIDAK FOKUS"
                color = (0, 255, 0) if prediction > 0.5 else (0, 0, 255)
        
        # Overlays
        cv2.putText(frame, f"Predict: {current_prediction}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(frame, f"Conf: {confidence:.2f}", (30, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Write Frame
        out.write(frame)
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    print(f"\nProcessing complete. Result saved at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Focus Estimation Video Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, help="Path to save output video file")
    
    args = parser.parse_args()
    
    input_video = args.input
    output_video = args.output if args.output else input_video.replace(".mp4", "_processed.mp4")
    
    process_video(input_video, output_video)
