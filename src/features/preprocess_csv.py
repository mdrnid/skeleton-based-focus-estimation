import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Pastikan project root ada di sys.path agar import src.* bisa berjalan
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.augmentation import apply_jitter, apply_scaling, apply_horizontal_flip

# ============================================================
#  KONFIGURASI
# ============================================================
SEQUENCE_LENGTH = 20   
STRIDE          = 5    # Geser jendela setiap 5 frame
N_LANDMARKS     = 17
RAW_FEATURES    = N_LANDMARKS * 4   
ENG_FEATURES    = 2                 
TOTAL_FEATURES  = RAW_FEATURES + ENG_FEATURES  

# Mapping subclass behavior (6 kelas)
SUBCLASS_MAP = {
    "melihat_layar":      0,
    "membaca_materi":     1,
    "menulis":            2,
    "menggunakan_ponsel": 3,
    "menoleh":            4,
    "tidur":              5,
}

SUBCLASS_TO_PARENT = {
    0: "fokus",
    1: "fokus",
    2: "fokus",
    3: "tidak_fokus",
    4: "tidak_fokus",
    5: "tidak_fokus",
}

BEHAVIOR_KEYWORDS = [
    "menggunakan_ponsel",
    "melihat_layar",
    "membaca_materi",
    "menoleh",
    "menulis",
    "tidur",
]

def normalize_frame(coords_reshaped):
    """Normalisasi satu frame landmark (shape: 17x4)."""
    c = coords_reshaped.copy()   
    
    # 1. Centering — Mid-Shoulder
    mid_shoulder = (c[11, 0:3] + c[12, 0:3]) / 2.0
    c[:, 0:3] -= mid_shoulder

    # 2. Scaling — Lebar Bahu
    shoulder_width = np.linalg.norm(c[11, 0:3] - c[12, 0:3])
    if shoulder_width > 1e-6:
        c[:, 0:3] /= shoulder_width

    # 3. Feature Engineering
    nose  = c[0, 0:2]
    l_ear = c[3, 0:2]
    r_ear = c[6, 0:2]

    dist_l = np.linalg.norm(nose - l_ear)
    dist_r = np.linalg.norm(nose - r_ear)

    ear_ratio     = (dist_l / dist_r) if dist_r > 1e-6 else 1.0
    nose_offset_x = c[0, 0]

    flat_raw = c.flatten()
    engineered = np.array([ear_ratio, nose_offset_x], dtype=np.float32)

    return np.concatenate([flat_raw, engineered])

def extract_behavior(video_id):
    """Ekstrak keyword behavior dari nama file video."""
    vid_lower = video_id.lower()
    for keyword in BEHAVIOR_KEYWORDS:
        if keyword in vid_lower:
            return keyword
    return None

def preprocess_pose_csv(input_csv, output_dir, window_size=SEQUENCE_LENGTH, stride=STRIDE):
    print(f"Membaca data dari {input_csv}...")
    df = pd.read_csv(input_csv)

    print("Mapping subclass labels...")
    df['behavior'] = df['video_id'].apply(extract_behavior)
    df = df[df['behavior'].notna()].copy()
    df['label_idx'] = df['behavior'].map(SUBCLASS_MAP).astype(int)

    landmark_cols = []
    for i in range(N_LANDMARKS):
        landmark_cols.extend([f'x_{i}', f'y_{i}', f'z_{i}', f'v_{i}'])

    processed_sequences = []
    labels              = []

    video_groups = df.groupby('video_id')
    print(f"Memproses {len(video_groups)} video dengan Sliding Window (size={window_size}, stride={stride})...")

    for vid_id, group in video_groups:
        group = group.sort_values('frame_num')
        label = group['label_idx'].iloc[0]

        if len(group) < window_size:
            continue

        # 1. Normalisasi semua frame dalam video
        video_features = []
        for _, row in group.iterrows():
            coords = row[landmark_cols].values.astype(np.float32).reshape(N_LANDMARKS, 4)
            video_features.append(normalize_frame(coords))
        
        video_features = np.array(video_features) # (T, 70)

        # 2. Ambil SATU Window (Jendela) saja di tengah video biar lebih representatif
        num_frames = len(video_features)
        middle_idx = (num_frames - window_size) // 2
        
        if middle_idx < 0: # Backup jika video sangat pendek
            middle_idx = 0
            
        window = video_features[middle_idx : middle_idx + window_size]
        
        # 3. Augmentation (Tetap 6x variasi buat bantu model belajar)
        flipped = apply_horizontal_flip(window)
        
        augmented_variants = [
            window,                                     # Original
            apply_jitter(window, sigma=0.003),          # + Jitter
            apply_scaling(window, scale_range=(0.98, 1.02)), # + Scale
            flipped,                                    # Flip
            apply_jitter(flipped, sigma=0.003),         # Flip + Jitter
            apply_scaling(flipped, scale_range=(0.98, 1.02)) # Flip + Scale
        ]

        for variant in augmented_variants:
            processed_sequences.append(variant)
            labels.append(label)

    X = np.array(processed_sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)

    print(f"\n[OK] Preprocessing Selesai!")
    print(f"   Total Sampel : {X.shape[0]}")
    print(f"   Shape X      : {X.shape}  -> [Samples, Window, Features]")
    print(f"   Shape y      : {y.shape}")
    
    inv_map = {v: k for k, v in SUBCLASS_MAP.items()}
    for cls_idx in sorted(inv_map.keys()):
        count = (y == cls_idx).sum()
        print(f"     [{cls_idx}] {inv_map[cls_idx]:20s} : {count} sampel")
    
    print(f"   Output saved to: {output_dir}")

if __name__ == "__main__":
    BASE_DIR  = Path(__file__).resolve().parent.parent.parent
    INPUT_CSV = BASE_DIR / "data" / "extractions" / "dataset_full.csv"
    OUTPUT_FOLDER = BASE_DIR / "data" / "processed"

    preprocess_pose_csv(str(INPUT_CSV), str(OUTPUT_FOLDER))
