"""
inference_utils.py
Shared utility functions for both live_inference.py and video_inference.py.
Fungsi di sini HARUS identik dengan logika di src/features/preprocess_csv.py
agar tidak ada inkonsistensi antara training dan inference.
"""

import numpy as np

# ============================================================
#  Konstanta (harus sinkron dengan preprocess_csv_minimal.py)
# ============================================================
SEQUENCE_LENGTH = 45
N_LANDMARKS     = 17
TOTAL_FEATURES  = 68   # HANYA Raw Landmarks (17*4)
NUM_CLASSES     = 6

# Label mapping
SUBCLASS_NAMES = {
    0: "Melihat Layar",
    1: "Membaca Materi",
    2: "Menulis",
    3: "Menggunakan Ponsel",
    4: "Menoleh",
    5: "Tidur",
}

SUBCLASS_TO_PARENT = {
    0: "FOKUS",
    1: "FOKUS",
    2: "FOKUS",
    3: "TIDAK FOKUS",
    4: "TIDAK FOKUS",
    5: "TIDAK FOKUS",
}

FOKUS_CLASSES     = {0, 1, 2}
TIDAK_FOKUS_CLASSES = {3, 4, 5}


def normalize_frame(coords_reshaped):
    """
    Normalisasi Minimal: Centering di hidung (landmark 0).
    Sesuai dengan logic di preprocess_csv_minimal.py.
    """
    c = coords_reshaped.copy()   # (17, 4)  [x, y, z, v]

    # --- 1. CENTERING: Hidung (Landmark 0) ---
    nose_coord = c[0, 0:3]
    c[:, 0:3] -= nose_coord

    # Tanpa Feature Engineering tambahan (Brutal-free)
    return c.flatten()                 # (68,)


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
        np.ndarray shape (1, SEQUENCE_LENGTH, TOTAL_FEATURES)  -- siap masuk model.
    """
    normalized = normalize_landmarks_buffer(landmarks_buffer)
    resampled  = resample_sequence(normalized, SEQUENCE_LENGTH)
    return resampled.reshape(1, SEQUENCE_LENGTH, TOTAL_FEATURES)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def decode_prediction(prediction_probs):
    """
    Decode output (logits/probs) dari model multi-class.

    Args:
        prediction_probs: np.ndarray shape (6,) — output raw dari model.

    Returns:
        dict dengan keys:
          - class_idx:   int, indeks kelas prediksi
          - subclass:    str, nama subclass (e.g. "Menoleh")
          - parent:      str, "FOKUS" atau "TIDAK FOKUS"
          - confidence:  float, probabilitas (0.0 - 1.0)
          - is_fokus:    bool
    """
    # Jika output adalah logits (ada nilai > 1 atau < 0), terapkan softmax
    if np.max(prediction_probs) > 1.0 or np.min(prediction_probs) < 0.0:
        probs = softmax(prediction_probs)
    else:
        probs = prediction_probs

    class_idx  = int(np.argmax(probs))
    confidence = float(probs[class_idx])
    subclass   = SUBCLASS_NAMES[class_idx]
    parent     = SUBCLASS_TO_PARENT[class_idx]
    is_fokus   = class_idx in FOKUS_CLASSES

    return {
        "class_idx":  class_idx,
        "subclass":   subclass,
        "parent":     parent,
        "confidence": confidence,
        "is_fokus":   is_fokus,
    }
