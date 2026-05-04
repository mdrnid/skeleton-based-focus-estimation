"""
holistic_config.py
==================
Central configuration for the MediaPipe Holistic extraction pipeline.

All landmark index definitions, feature counts, and label mappings live here
so that extract_holistic.py, preprocess_holistic.py, and live_inference can
share a single source of truth.
"""

# ============================================================
#  SEQUENCE / WINDOW PARAMETERS
# ============================================================
SEQUENCE_LENGTH = 45   # Window size for CNN-LSTM input
STRIDE          = 5    # Sliding window step (for training data)

# ============================================================
#  FACE MESH — SELECTED LANDMARK INDICES  (18 out of 468)
# ============================================================
#
# MediaPipe Face Mesh canonical landmark indices:
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
#
# --- Eyes (4 pts) ---
#   33  : Right eye outer corner
#   133 : Right eye inner corner
#   362 : Left eye outer corner
#   263 : Left eye inner corner
#
# --- Pupils / Iris center approximation (2 pts) ---
#   468 : Right iris center  (available when refine_face_landmarks=True)
#   473 : Left iris center   (available when refine_face_landmarks=True)
#   NOTE: If iris landmarks are unavailable, fall back to pupil proxies:
#         159 (right eye top lid center), 386 (left eye top lid center).
#
# --- Nose (1 pt) ---
#   1   : Nose tip
#
# --- Eyebrows (4 pts) ---
#   70  : Right eyebrow inner
#   105 : Right eyebrow outer
#   300 : Left eyebrow inner
#   334 : Left eyebrow outer
#
# --- Jaw / Face silhouette for head pose estimation (5 pts) ---
#   10  : Top of forehead (mid-sagittal)
#   152 : Bottom of chin
#   234 : Right cheek (ear level)
#   454 : Left cheek (ear level)
#   168 : Nose bridge (between eyes) — useful for pitch calculation
#
FACE_SELECTED_INDICES = [
    # Eyes — corners
    33, 133,    # right eye outer, inner
    362, 263,   # left  eye outer, inner
    # Pupils / iris center (requires refine_face_landmarks=True)
    468, 473,
    # Nose tip
    1,
    # Eyebrows
    70, 105,    # right eyebrow inner, outer
    300, 334,   # left  eyebrow inner, outer
    # Jaw / Silhouette — for head pitch / yaw / roll
    10,         # forehead top
    152,        # chin bottom
    234,        # right cheek
    454,        # left  cheek
    168,        # nose bridge (between eyes)
]

# We also define the nose tip index WITHIN FACE_SELECTED_INDICES
# so normalization can look it up cheaply.
FACE_NOSE_TIP_LOCAL_IDX = FACE_SELECTED_INDICES.index(1)   # → 6

# Fallback pupil indices when iris landmarks (468, 473) are not available
FACE_PUPIL_FALLBACK = {468: 159, 473: 386}

# ============================================================
#  POSE — ALL 33 LANDMARKS
# ============================================================
N_POSE_LANDMARKS = 33

# Key pose landmark indices (MediaPipe Pose canonical):
POSE_LEFT_SHOULDER  = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_HIP       = 23
POSE_RIGHT_HIP      = 24

# ============================================================
#  HANDS — ALL 21 LANDMARKS EACH
# ============================================================
N_HAND_LANDMARKS = 21
HAND_WRIST_IDX   = 0    # Wrist is always index 0 in hand landmarks

# ============================================================
#  FEATURE COUNTS  (x, y, z per landmark — NO visibility)
# ============================================================
N_FACE_SELECTED = len(FACE_SELECTED_INDICES)        # 18
N_FEATURES_FACE = N_FACE_SELECTED * 3               # 54
N_FEATURES_POSE = N_POSE_LANDMARKS * 3              # 99
N_FEATURES_HAND = N_HAND_LANDMARKS * 3              # 63  (per hand)

TOTAL_FEATURES_PER_FRAME = (
    N_FEATURES_FACE
    + N_FEATURES_POSE
    + N_FEATURES_HAND   # left hand
    + N_FEATURES_HAND   # right hand
)
# 54 + 99 + 63 + 63 = 279 features per frame

# ============================================================
#  LABEL MAPPING  (6 behaviour classes)
# ============================================================
SUBCLASS_MAP = {
    "melihat_layar":      0,
    "membaca_materi":     1,
    "menulis":            2,
    "menggunakan_ponsel": 3,
    "menoleh":            4,
    "tidur":              5,
}

SUBCLASS_NAMES = {v: k for k, v in SUBCLASS_MAP.items()}

SUBCLASS_TO_PARENT = {
    0: "FOKUS",
    1: "FOKUS",
    2: "FOKUS",
    3: "TIDAK FOKUS",
    4: "TIDAK FOKUS",
    5: "TIDAK FOKUS",
}

FOKUS_CLASSES       = {0, 1, 2}
TIDAK_FOKUS_CLASSES = {3, 4, 5}

NUM_CLASSES = len(SUBCLASS_MAP)

# ============================================================
#  BEHAVIOR KEYWORD DETECTION  (for filename → label mapping)
# ============================================================
BEHAVIOR_KEYWORDS = [
    "menggunakan_ponsel",
    "melihat_layar",
    "membaca_materi",
    "menoleh",
    "menulis",
    "tidur",
]
