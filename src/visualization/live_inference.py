import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from pathlib import Path
import time

# --- FIX FOR KERAS 3 COMPATIBILITY ERROR ---
# This patches the Dense layer to ignore 'quantization_config' if present
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
BUFFER_SIZE = 20  # Maintain a buffer of 20 frames to resample to 10
THRESHOLD = 0.5   # Focus/Not Focus threshold

# --- PREPROCESSING UTILS ---
def resample_sequence(frames_features, target_length):
    """Interpolates temporal sequence to exactly target_length frames."""
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
    """
    Apply spatial normalization (centering at nose, scaling by shoulder width)
    landmarks_buffer: List of 17 landmarks per frame
    """
    frames_features = []
    
    for landmarks in landmarks_buffer:
        # Landmarks is a list of 17 [x, y, z, v]
        coords = np.array(landmarks, dtype=np.float32).reshape(17, 4)
        
        # 1. Centering (Relative to Nose / Landmark 0)
        nose = coords[0, 0:3]
        coords[:, 0:3] -= nose
        
        # 2. Scaling (Based on shoulder width / Landmarks 11 & 12)
        l_shoulder = coords[11, 0:3]
        r_shoulder = coords[12, 0:3]
        shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)
        
        if shoulder_width > 1e-6:
            coords[:, 0:3] /= shoulder_width
            
        frames_features.append(coords.flatten())
        
    return np.array(frames_features)

def draw_landmarks(frame, results):
    """Helper to draw basic skeleton for visualization."""
    if not results.pose_landmarks:
        return
    
    # Simple connections for the first 17 landmarks
    connections = [
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), # Face
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Upper Body
        (11, 12) # Shoulders
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
def main():
    print("Initializing inference system...")
    
    # 1. Load Keras Model
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("Keras model loaded.")

    # 2. Setup MediaPipe Landmarker
    if not POSE_TASK_PATH.exists():
        print(f"Error: Pose task model not found at {POSE_TASK_PATH}")
        return
        
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

    # 3. Setup Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Buffer to hold landmarks for the sequence
    landmarks_buffer = []
    current_prediction = "Waiting..."
    confidence = 0.0

    print("--- LIVE INFERENCE STARTED ---")
    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect Landmarks
        results = landmarker.detect(mp_image)

        if results.pose_landmarks:
            # Extract 17 landmarks
            frame_landmarks = []
            for landmark in results.pose_landmarks[0][:17]:
                frame_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            
            landmarks_buffer.append(frame_landmarks)
            
            # Keep buffer size
            if len(landmarks_buffer) > BUFFER_SIZE:
                landmarks_buffer.pop(0)

            # Draw skeleton
            draw_landmarks(frame, results)
        else:
            # If no detection, we could add zeros or just skip/wait
            # For robustness in inference, we might want to clear buffer if detection is lost for too long
            # but for now, we just skip updating the buffer
            pass

        # Perform Inference if buffer is full enough
        if len(landmarks_buffer) >= 5: # Minimum frames to start inferencing
            # Preprocess
            features_raw = np.array(landmarks_buffer)
            # 1. Normalize
            features_normalized = normalize_landmarks(landmarks_buffer)
            # 2. Resample to 10 frames
            features_resampled = resample_sequence(features_normalized, SEQUENCE_LENGTH)
            
            # 3. Model Predict
            # Shape: (1, 10, 68)
            input_data = features_resampled.reshape(1, SEQUENCE_LENGTH, 68)
            prediction = model.predict(input_data, verbose=0)[0][0]
            
            confidence = prediction if prediction > 0.5 else 1.0 - prediction
            current_prediction = "FOCUS" if prediction > 0.5 else "TIDAK FOKUS"
            
            # Color logic
            color = (0, 255, 0) if prediction > 0.5 else (0, 0, 255)
        else:
            color = (255, 255, 255)

        # UI Overlays
        cv2.putText(frame, f"Status: {current_prediction}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Conf: {confidence:.2f}", (30, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Buffer: {len(landmarks_buffer)}/{BUFFER_SIZE}", (30, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Focus Estimation Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Inference ended.")

if __name__ == "__main__":
    main()
