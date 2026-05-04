"""
preprocess_holistic.py
=======================
Reads the CSV produced by extract_holistic.py, applies:
  1. Relative normalization (Constraint 3)
     • Face  → relative to nose tip
     • Hands → relative to wrist
     • Pose  → relative to mid-shoulder
  2. Sliding-window segmentation (Window = 45, Stride = 5)
  3. Saves X.npy (samples, 45, 279) and y.npy ready for CNN-LSTM.

Usage:
    python preprocess_holistic.py
    (configure paths at the bottom or import functions directly)
"""

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.holistic_config import (
    SEQUENCE_LENGTH,
    STRIDE,
    N_FACE_SELECTED,
    N_POSE_LANDMARKS,
    N_HAND_LANDMARKS,
    FACE_NOSE_TIP_LOCAL_IDX,
    HAND_WRIST_IDX,
    POSE_LEFT_SHOULDER,
    POSE_RIGHT_SHOULDER,
    TOTAL_FEATURES_PER_FRAME,
    SUBCLASS_MAP,
    BEHAVIOR_KEYWORDS,
)


# ============================================================
#  NORMALIZATION  (Constraint 3 — Relative / Geometric)
# ============================================================

def normalize_face(face_coords):
    """
    Normalize face landmarks relative to the nose tip.

    Args:
        face_coords: np.ndarray (N_FACE_SELECTED, 3)

    Returns:
        np.ndarray (N_FACE_SELECTED, 3) — all coordinates centered on nose tip.
    """
    c = face_coords.copy()
    nose = c[FACE_NOSE_TIP_LOCAL_IDX].copy()   # (3,) — nose tip
    c -= nose  # broadcast: every point becomes (x - x_nose, y - y_nose, z - z_nose)
    return c


def normalize_hand(hand_coords):
    """
    Normalize hand landmarks relative to the wrist (landmark 0).

    Args:
        hand_coords: np.ndarray (21, 3)

    Returns:
        np.ndarray (21, 3) — all coordinates centered on the wrist.
    """
    c = hand_coords.copy()
    wrist = c[HAND_WRIST_IDX].copy()   # (3,)
    c -= wrist
    return c


def normalize_pose(pose_coords):
    """
    Normalize pose landmarks relative to the mid-shoulder.
    mid_shoulder = (left_shoulder + right_shoulder) / 2

    Args:
        pose_coords: np.ndarray (33, 3)

    Returns:
        np.ndarray (33, 3) — all coordinates centered on mid-shoulder.
    """
    c = pose_coords.copy()
    left_sh  = c[POSE_LEFT_SHOULDER].copy()
    right_sh = c[POSE_RIGHT_SHOULDER].copy()
    mid_shoulder = (left_sh + right_sh) / 2.0
    c -= mid_shoulder
    return c


def normalize_frame_holistic(face, pose, left_hand, right_hand):
    """
    Apply split relative normalization to one frame and return a flat vector.

    Args:
        face:       (N_FACE_SELECTED, 3)
        pose:       (33, 3)
        left_hand:  (21, 3)
        right_hand: (21, 3)

    Returns:
        np.ndarray (TOTAL_FEATURES_PER_FRAME,) — flat 1D vector (279,)
    """
    face_n = normalize_face(face)
    pose_n = normalize_pose(pose)
    lh_n   = normalize_hand(left_hand)
    rh_n   = normalize_hand(right_hand)

    return np.concatenate([
        face_n.flatten(),
        pose_n.flatten(),
        lh_n.flatten(),
        rh_n.flatten(),
    ]).astype(np.float32)


# ============================================================
#  CSV PARSING HELPERS
# ============================================================

def _parse_landmarks_from_row(row):
    """
    Parse a single CSV row into four (N, 3) arrays.

    Column layout (set by extract_holistic.py):
      face_x_0 .. face_z_{N-1}  |  pose_x_0 .. pose_z_32  |  lh_x_0 .. lh_z_20  |  rh_x_0 .. rh_z_20
    """
    face_cols = []
    for i in range(N_FACE_SELECTED):
        face_cols.extend([f"face_x_{i}", f"face_y_{i}", f"face_z_{i}"])

    pose_cols = []
    for i in range(N_POSE_LANDMARKS):
        pose_cols.extend([f"pose_x_{i}", f"pose_y_{i}", f"pose_z_{i}"])

    lh_cols = []
    for i in range(N_HAND_LANDMARKS):
        lh_cols.extend([f"lh_x_{i}", f"lh_y_{i}", f"lh_z_{i}"])

    rh_cols = []
    for i in range(N_HAND_LANDMARKS):
        rh_cols.extend([f"rh_x_{i}", f"rh_y_{i}", f"rh_z_{i}"])

    face = row[face_cols].values.astype(np.float32).reshape(N_FACE_SELECTED, 3)
    pose = row[pose_cols].values.astype(np.float32).reshape(N_POSE_LANDMARKS, 3)
    lh   = row[lh_cols].values.astype(np.float32).reshape(N_HAND_LANDMARKS, 3)
    rh   = row[rh_cols].values.astype(np.float32).reshape(N_HAND_LANDMARKS, 3)

    return face, pose, lh, rh


def _build_landmark_col_names():
    """Build the flat list of landmark column names (for fast vectorized access)."""
    cols = []
    for prefix, count in [("face", N_FACE_SELECTED),
                          ("pose", N_POSE_LANDMARKS),
                          ("lh",   N_HAND_LANDMARKS),
                          ("rh",   N_HAND_LANDMARKS)]:
        for i in range(count):
            cols.extend([f"{prefix}_x_{i}", f"{prefix}_y_{i}", f"{prefix}_z_{i}"])
    return cols


# ============================================================
#  RESAMPLING  (for short videos < SEQUENCE_LENGTH)
# ============================================================

def resample_sequence(frames, target_length):
    """
    Linearly interpolate a sequence of feature vectors to a target length.

    Args:
        frames: np.ndarray (T, F)
        target_length: int

    Returns:
        np.ndarray (target_length, F)
    """
    n_frames, n_features = frames.shape
    if n_frames == target_length:
        return frames

    src_idx    = np.arange(n_frames)
    target_idx = np.linspace(0, n_frames - 1, target_length)

    resampled = np.zeros((target_length, n_features), dtype=np.float32)
    for i in range(n_features):
        resampled[:, i] = np.interp(target_idx, src_idx, frames[:, i])
    return resampled


# ============================================================
#  LABEL EXTRACTION  (from filename)
# ============================================================

def _extract_behavior(video_id):
    """Extract behavior keyword from video filename."""
    vid_lower = video_id.lower()
    for kw in BEHAVIOR_KEYWORDS:
        if kw in vid_lower:
            return kw
    return None


# ============================================================
#  MAIN PREPROCESSING PIPELINE
# ============================================================

def preprocess_holistic(input_csv, output_dir,
                        window_size=SEQUENCE_LENGTH, stride=STRIDE):
    """
    Full preprocessing pipeline:
      1. Read CSV from extract_holistic.py
      2. Map labels from filenames
      3. Per-frame: parse → normalize (split relative) → flatten
      4. Per-video: sliding window (or resample if short)
      5. Save X.npy, y.npy

    Args:
        input_csv:   Path to the holistic CSV.
        output_dir:  Directory to save X.npy and y.npy.
        window_size: Sequence length (default 45).
        stride:      Sliding window step (default 5).
    """
    print(f"Reading CSV: {input_csv} ...")
    df = pd.read_csv(input_csv)
    print(f"  Rows loaded: {len(df)}")

    # --- Label mapping ---
    df["behavior"] = df["video_id"].apply(_extract_behavior)
    df = df[df["behavior"].notna()].copy()
    df["label_idx"] = df["behavior"].map(SUBCLASS_MAP).astype(int)

    # Pre-build column name lists for fast access
    landmark_cols = _build_landmark_col_names()

    sequences = []
    labels    = []

    video_groups = df.groupby("video_id")
    print(f"Processing {len(video_groups)} videos ...")

    for vid_id, group in video_groups:
        group = group.sort_values("frame_num")
        label = group["label_idx"].iloc[0]

        # --- Per-frame normalization ---
        video_features = []
        for _, row in group.iterrows():
            face, pose, lh, rh = _parse_landmarks_from_row(row)
            feat_vec = normalize_frame_holistic(face, pose, lh, rh)
            video_features.append(feat_vec)

        video_features = np.array(video_features, dtype=np.float32)  # (T, 279)
        num_frames = len(video_features)

        # --- Resample if too short ---
        if num_frames < window_size:
            video_features = resample_sequence(video_features, window_size)
            num_frames = window_size

        # --- Sliding Window ---
        for start in range(0, num_frames - window_size + 1, stride):
            window = video_features[start : start + window_size]
            sequences.append(window)
            labels.append(label)

    X = np.array(sequences, dtype=np.float32)   # (N, 45, 279)
    y = np.array(labels, dtype=np.int32)         # (N,)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)

    # --- Summary ---
    print(f"\n{'='*55}")
    print(f"  Preprocessing Complete!")
    print(f"  X shape : {X.shape}  →  [Samples, {window_size}, {TOTAL_FEATURES_PER_FRAME}]")
    print(f"  y shape : {y.shape}")
    print(f"{'='*55}")

    inv_map = {v: k for k, v in SUBCLASS_MAP.items()}
    for idx in sorted(inv_map):
        count = int((y == idx).sum())
        print(f"    [{idx}] {inv_map[idx]:25s} : {count} samples")

    print(f"\n  Saved to: {output_dir}")
    print(f"{'='*55}")


# ============================================================
#  REAL-TIME (LIVE) EXTRACTION + NORMALIZATION
# ============================================================
#  These functions are designed for live_inference.py so it can
#  call them frame-by-frame without touching CSV files.

def extract_and_normalize_live(results, previous_pose=None):
    """
    One-shot: extract + normalize a single frame from Holistic results.
    Returns the flat (279,) feature vector and the raw pose for caching.

    This function is the bridge between MediaPipe output and model input
    during live inference.

    Args:
        results:       MediaPipe Holistic results object.
        previous_pose: np.ndarray (33, 3) from the previous frame,
                       used as hand fallback if the current pose is missing.

    Returns:
        tuple: (flat_features, pose_coords)
            flat_features: np.ndarray (279,)
            pose_coords:   np.ndarray (33, 3) — cache this for the next call
    """
    # Import here to avoid circular dependency
    from src.features.extract_holistic import extract_frame_holistic

    face, pose, lh, rh = extract_frame_holistic(results, previous_pose)
    flat = normalize_frame_holistic(face, pose, lh, rh)
    return flat, pose


def prepare_model_input_holistic(landmarks_buffer):
    """
    Convert a list of flat feature vectors into a batch-1 tensor for the model.

    Args:
        landmarks_buffer: list of np.ndarray (279,), length >= SEQUENCE_LENGTH

    Returns:
        np.ndarray (1, SEQUENCE_LENGTH, 279) — ready for model.predict()
    """
    arr = np.array(landmarks_buffer[-SEQUENCE_LENGTH:], dtype=np.float32)

    if arr.shape[0] < SEQUENCE_LENGTH:
        arr = resample_sequence(arr, SEQUENCE_LENGTH)

    return arr.reshape(1, SEQUENCE_LENGTH, TOTAL_FEATURES_PER_FRAME)


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    BASE_DIR    = _PROJECT_ROOT
    INPUT_CSV   = BASE_DIR / "data" / "extractions" / "holistic_raw.csv"
    OUTPUT_DIR  = BASE_DIR / "data" / "processed_holistic"

    preprocess_holistic(str(INPUT_CSV), str(OUTPUT_DIR))
