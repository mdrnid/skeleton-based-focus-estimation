"""
extract_holistic.py
====================
Extracts landmarks from video files or webcam using MediaPipe Tasks (HolisticLandmarker).
Compatible with MediaPipe 0.10.x+.

Constraints Implemented:
  1. Feature Selection: Only ~16-20 crucial face landmarks.
  2. Missing Value Handling: Hands fall back to Hip/Shoulder coordinates from Pose.
  3. Output: Flattened (x, y, z) coordinates per frame.

Usage:
    python extract_holistic.py <dataset_path> <output_csv> [stride]
    python extract_holistic.py --webcam <output_csv>
"""

import cv2
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------
#  Project Imports
# ---------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.holistic_config import (
    FACE_SELECTED_INDICES,
    FACE_PUPIL_FALLBACK,
    N_FACE_SELECTED,
    N_POSE_LANDMARKS,
    N_HAND_LANDMARKS,
    POSE_LEFT_SHOULDER,
    POSE_RIGHT_SHOULDER,
    POSE_LEFT_HIP,
    POSE_RIGHT_HIP,
    BEHAVIOR_KEYWORDS,
)

# ============================================================
#  CORE EXTRACTION LOGIC
# ============================================================

def _extract_list_landmarks(landmark_list, n_expected, selected_indices=None):
    """Helper to convert list of NormalizedLandmarks to numpy (N, 3)."""
    if not landmark_list or len(landmark_list) == 0:
        return None
    
    # In Holistic Tasks API, these are direct lists of landmarks for one person
    lms = landmark_list
    
    if selected_indices:
        coords = np.zeros((len(selected_indices), 3), dtype=np.float32)
        for local_idx, global_idx in enumerate(selected_indices):
            if global_idx < len(lms):
                lm = lms[global_idx]
                coords[local_idx] = [lm.x, lm.y, lm.z]
            else:
                # Fallback for iris if refine_face_landmarks didn't catch them
                fallback = FACE_PUPIL_FALLBACK.get(global_idx, 1) # default nose tip
                lm = lms[fallback]
                coords[local_idx] = [lm.x, lm.y, lm.z]
        return coords
    else:
        coords = np.zeros((n_expected, 3), dtype=np.float32)
        for i in range(min(len(lms), n_expected)):
            lm = lms[i]
            coords[i] = [lm.x, lm.y, lm.z]
        return coords

def _hand_fallback_from_pose(pose_coords, side="left"):
    """Constraint 2: Hip/Chest fallback for undetected hands."""
    if pose_coords is None:
        return np.zeros((N_HAND_LANDMARKS, 3), dtype=np.float32)

    hip_idx = POSE_LEFT_HIP if side == "left" else POSE_RIGHT_HIP
    fallback_coord = pose_coords[hip_idx].copy()

    # If hip is 0 (not visible), use mid-shoulder
    if np.allclose(fallback_coord, 0.0):
        ls = pose_coords[POSE_LEFT_SHOULDER]
        rs = pose_coords[POSE_RIGHT_SHOULDER]
        fallback_coord = (ls + rs) / 2.0

    return np.tile(fallback_coord, (N_HAND_LANDMARKS, 1)).astype(np.float32)

def extract_frame_holistic(result, prev_pose=None):
    """Parses HolisticLandmarkerResult into normalized numpy arrays."""
    # Pose
    pose = _extract_list_landmarks(result.pose_landmarks, N_POSE_LANDMARKS)
    pose_for_fallback = pose if pose is not None else prev_pose
    
    # Face (Selected Indices)
    face = _extract_list_landmarks(result.face_landmarks, N_FACE_SELECTED, FACE_SELECTED_INDICES)
    if face is None: face = np.zeros((N_FACE_SELECTED, 3), dtype=np.float32)
    
    # Pose Final (Zero fill if still None)
    if pose is None: pose = np.zeros((N_POSE_LANDMARKS, 3), dtype=np.float32)
    
    # Hands
    lh = _extract_list_landmarks(result.left_hand_landmarks, N_HAND_LANDMARKS)
    if lh is None: lh = _hand_fallback_from_pose(pose_for_fallback, "left")
    
    rh = _extract_list_landmarks(result.right_hand_landmarks, N_HAND_LANDMARKS)
    if rh is None: rh = _hand_fallback_from_pose(pose_for_fallback, "right")
    
    return face, pose, lh, rh

def flatten_frame(face, pose, lh, rh):
    """Flattens landmarks into a 1D vector (N_Features * 3)."""
    return np.concatenate([face.flatten(), pose.flatten(), lh.flatten(), rh.flatten()])

# ============================================================
#  PROCESSING PIPELINES
# ============================================================

def get_holistic_landmarker(model_path):
    """Initializes the MediaPipe HolisticLandmarker Task."""
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5
    )
    return vision.HolisticLandmarker.create_from_options(options)

def process_videos(dataset_path, output_csv, model_path, stride=1):
    dataset_dir = Path(dataset_path)
    video_files = sorted(list(dataset_dir.rglob('*.mp4')))
    
    if not video_files:
        print(f"No videos found in {dataset_path}")
        return

    landmarker = get_holistic_landmarker(model_path)
    data = []
    
    # Column Headers
    cols = ['video_id', 'frame_num', 'main_label', 'sub_label']
    for p, n in [('face', N_FACE_SELECTED), ('pose', N_POSE_LANDMARKS), ('lh', N_HAND_LANDMARKS), ('rh', N_HAND_LANDMARKS)]:
        for i in range(n): cols.extend([f'{p}_x_{i}', f'{p}_y_{i}', f'{p}_z_{i}'])

    for video_path in tqdm(video_files, desc="Processing Videos"):
        video_id = video_path.name
        sub_label = video_path.parent.name
        main_label = "fokus" if "fokus" in str(video_path).lower() and "tidak_fokus" not in str(video_path).lower() else "tidak_fokus"
        
        cap = cv2.VideoCapture(str(video_path))
        frame_num = 0
        prev_pose = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_num % stride == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                result = landmarker.detect(mp_image)
                face, pose, lh, rh = extract_frame_holistic(result, prev_pose)
                prev_pose = pose
                
                flat = flatten_frame(face, pose, lh, rh)
                data.append([video_id, frame_num, main_label, sub_label] + flat.tolist())
                
            frame_num += 1
        cap.release()

    df = pd.DataFrame(data, columns=cols)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Extraction complete! Saved to {output_csv}")

if __name__ == "__main__":
    MODEL_PATH = str(_PROJECT_ROOT / "models" / "holistic_landmarker.task")
    
    if len(sys.argv) < 3:
        print("Usage: python extract_holistic.py <dataset_path> <output_csv> [stride]")
        sys.exit(1)
        
    process_videos(sys.argv[1], sys.argv[2], MODEL_PATH, int(sys.argv[3]) if len(sys.argv) > 3 else 1)
