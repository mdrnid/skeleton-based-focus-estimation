"""
inference_utils.py
Shared utility functions for both live_inference.py and video_inference.py.
Fungsi di sini HARUS identik dengan logika di src/features/preprocess_csv.py
agar tidak ada inkonsistensi antara training dan inference.
"""

import numpy as np

# ============================================================
#  Konstanta (harus sinkron dengan preprocess_csv.py)
# ============================================================
SEQUENCE_LENGTH = 20
N_LANDMARKS     = 17
TOTAL_FEATURES  = 70   # 17*4 (raw) + 2 (engineered)


def normalize_frame(coords_reshaped):
    """
    Normalisasi satu frame landmark (shape: 17x4).
    IDENTIK dengan normalize_frame() di preprocess_csv.py.

    Pipeline:
      1. Centering ke Mid-Shoulder (bukan hidung).
      2. Scaling dengan lebar bahu.
      3. Feature engineering: ear_ratio dan nose_offset_x.
    """
    c = coords_reshaped.copy()   # (17, 4)  [x, y, z, v]

    # --- 1. CENTERING: Mid-Shoulder ---
    mid_shoulder = (c[11, 0:3] + c[12, 0:3]) / 2.0
    c[:, 0:3] -= mid_shoulder

    # --- 2. SCALING: lebar bahu ---
    shoulder_width = np.linalg.norm(c[11, 0:3] - c[12, 0:3])
    if shoulder_width > 1e-6:
        c[:, 0:3] /= shoulder_width

    # --- 3. FEATURE ENGINEERING ---
    nose  = c[0, 0:2]
    l_ear = c[3, 0:2]
    r_ear = c[6, 0:2]

    dist_l = np.linalg.norm(nose - l_ear)
    dist_r = np.linalg.norm(nose - r_ear)

    ear_ratio     = (dist_l / dist_r) if dist_r > 1e-6 else 1.0
    nose_offset_x = c[0, 0]   # X hidung relatif mid-shoulder

    flat_raw   = c.flatten()                                      # (68,)
    engineered = np.array([ear_ratio, nose_offset_x], dtype=np.float32)  # (2,)

    return np.concatenate([flat_raw, engineered])                 # (70,)


def normalize_landmarks_buffer(landmarks_buffer):
    """
    Normalisasi seluruh buffer frame.

    Args:
        landmarks_buffer: List of lists, tiap elemen adalah 17*4=68 nilai
                          (x0, y0, z0, v0, x1, ...) dari MediaPipe.

    Returns:
        np.ndarray shape (T, 70)
    """
    frames_features = []
    for raw_lms in landmarks_buffer:
        coords = np.array(raw_lms, dtype=np.float32).reshape(N_LANDMARKS, 4)
        feat_vec = normalize_frame(coords)
        frames_features.append(feat_vec)
    return np.array(frames_features)


def resample_sequence(frames_features, target_length=SEQUENCE_LENGTH):
    """
    Resample temporal sequence ke panjang target.
    Input:  (T, F)
    Output: (target_length, F)
    """
    n_frames, n_features = frames_features.shape
    if n_frames == target_length:
        return frames_features

    src_idx    = np.arange(n_frames)
    target_idx = np.linspace(0, n_frames - 1, target_length)

    resampled = np.zeros((target_length, n_features))
    for i in range(n_features):
        resampled[:, i] = np.interp(target_idx, src_idx, frames_features[:, i])

    return resampled


def prepare_model_input(landmarks_buffer):
    """
    Full preprocessing pipeline untuk satu inference step.

    Args:
        landmarks_buffer: List of 68-element lists (raw MediaPipe normalized coords).

    Returns:
        np.ndarray shape (1, SEQUENCE_LENGTH, TOTAL_FEATURES)  — siap masuk model.
    """
    normalized = normalize_landmarks_buffer(landmarks_buffer)
    resampled  = resample_sequence(normalized, SEQUENCE_LENGTH)
    return resampled.reshape(1, SEQUENCE_LENGTH, TOTAL_FEATURES)
