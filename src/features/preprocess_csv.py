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
SEQUENCE_LENGTH = 20   # Ditingkatkan dari 10 → 20 agar temporal richer
N_LANDMARKS    = 17
RAW_FEATURES   = N_LANDMARKS * 4   # x, y, z, v per landmark = 68
ENG_FEATURES   = 2                 # ear_ratio, nose_offset_x
TOTAL_FEATURES = RAW_FEATURES + ENG_FEATURES  # 70


def resample_sequence(frames_features, target_length):
    """
    Melakukan interpolasi temporal (linear) untuk mengubah panjang sekuens
    dari N frame menjadi tepat target_length frame.
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


def normalize_frame(coords_reshaped):
    """
    Normalisasi satu frame landmark (shape: 17x4).

    Pipeline:
      1. Centering: pusat koordinat dipindah ke Mid-Shoulder (bukan hidung).
         Ini membuat gerakan kepala (menoleh, menunduk) terlihat jelas
         sebagai pergeseran koordinat hidung & telinga.
      2. Scaling: dibagi dengan lebar bahu agar invariant terhadap jarak kamera.
      3. Feature Engineering:
         - ear_ratio      : rasio jarak hidung-telinga kiri vs kanan (deteksi yaw).
         - nose_offset_x  : posisi X hidung relatif mid-shoulder (redundant tapi eksplisit).

    Returns:
      flat_features (np.ndarray): vektor (TOTAL_FEATURES,) = (70,)
    """
    # --- Salinan agar tidak memodifikasi array asli ---
    c = coords_reshaped.copy()   # shape (17, 4)

    # ---------------------------------------------------------------- #
    #  1. CENTERING — Mid-Shoulder sebagai titik pusat (0, 0, 0)
    #     Landmark 11 = bahu kiri, Landmark 12 = bahu kanan
    # ---------------------------------------------------------------- #
    mid_shoulder = (c[11, 0:3] + c[12, 0:3]) / 2.0
    c[:, 0:3] -= mid_shoulder

    # ---------------------------------------------------------------- #
    #  2. SCALING — Normalisasi skala dengan lebar bahu
    # ---------------------------------------------------------------- #
    l_shoulder    = c[11, 0:3]
    r_shoulder    = c[12, 0:3]
    shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)

    if shoulder_width > 1e-6:
        c[:, 0:3] /= shoulder_width

    # ---------------------------------------------------------------- #
    #  3. FEATURE ENGINEERING
    #     Landmark 3 = telinga kiri, Landmark 6 = telinga kanan (dalam 17 LM)
    #     Landmark 0 = hidung
    # ---------------------------------------------------------------- #
    nose     = c[0, 0:2]   # (x, y)
    l_ear    = c[3, 0:2]   # (x, y)
    r_ear    = c[6, 0:2]   # (x, y)

    dist_nose_l_ear = np.linalg.norm(nose - l_ear)
    dist_nose_r_ear = np.linalg.norm(nose - r_ear)

    # Ear ratio: > 1 → menoleh ke kanan, < 1 → menoleh ke kiri, ≈ 1 → lurus
    if dist_nose_r_ear > 1e-6:
        ear_ratio = dist_nose_l_ear / dist_nose_r_ear
    else:
        ear_ratio = 1.0

    # Posisi X hidung relatif mid-shoulder (sudah di-center, jadi ini langsung X hidung)
    nose_offset_x = c[0, 0]

    # ---------------------------------------------------------------- #
    #  Flatten + append engineered features
    # ---------------------------------------------------------------- #
    flat_raw = c.flatten()                                   # (68,)
    engineered = np.array([ear_ratio, nose_offset_x],
                          dtype=np.float32)                  # (2,)

    return np.concatenate([flat_raw, engineered])            # (70,)


def preprocess_pose_csv(input_csv, output_dir,
                        sequence_length=SEQUENCE_LENGTH,
                        min_frames=5):
    """
    Preprocessing data CSV landmark untuk siap masuk ke model LSTM.
    Menghasilkan X.npy (N, sequence_length, TOTAL_FEATURES) dan y.npy (N,).
    """
    print(f"Membaca data dari {input_csv}...")
    df = pd.read_csv(input_csv)

    # ---------------------------------------------------------------- #
    #  1. Label Mapping (1 = fokus, 0 = tidak_fokus)
    # ---------------------------------------------------------------- #
    print("Mapping labels...")
    label_map   = {"fokus": 1, "tidak_fokus": 0}
    df['label_idx'] = df['main_label'].map(label_map).fillna(-1).astype(int)
    df = df[df['label_idx'] != -1]

    # ---------------------------------------------------------------- #
    #  2. Kolom landmark
    # ---------------------------------------------------------------- #
    landmark_cols = []
    for i in range(N_LANDMARKS):
        landmark_cols.extend([f'x_{i}', f'y_{i}', f'z_{i}', f'v_{i}'])

    processed_data = []
    labels         = []

    video_groups = df.groupby('video_id')
    print(f"Memproses {len(video_groups)} video...")

    for vid_id, group in video_groups:
        group = group.sort_values('frame_num')
        label = group['label_idx'].iloc[0]

        if len(group) < min_frames:
            continue

        # ------------------------------------------------------------ #
        #  3. Normalisasi Spasial Per Frame (Mid-Shoulder Centered)
        # ------------------------------------------------------------ #
        frames_features = []
        for _, row in group.iterrows():
            coords = row[landmark_cols].values.astype(np.float32)
            coords_reshaped = coords.reshape(N_LANDMARKS, 4)
            feat_vec = normalize_frame(coords_reshaped)  # (70,)
            frames_features.append(feat_vec)

        frames_features = np.array(frames_features)  # (T, 70)

        # ------------------------------------------------------------ #
        #  4. Resampling Temporal → sequence_length frame
        # ------------------------------------------------------------ #
        frames_resampled = resample_sequence(frames_features, sequence_length)

        # ------------------------------------------------------------ #
        #  5. Augmentation (6x lipat data)
        # ------------------------------------------------------------ #
        flipped = apply_horizontal_flip(frames_resampled)

        augmented_variants = [
            frames_resampled,                                    # Original
            apply_jitter(frames_resampled, sigma=0.003),         # + Jitter
            apply_scaling(frames_resampled, scale_range=(0.98, 1.02)),  # + Scale
            flipped,                                             # Flip
            apply_jitter(flipped, sigma=0.003),                  # Flip + Jitter
            apply_scaling(flipped, scale_range=(0.98, 1.02)),    # Flip + Scale
        ]

        for variant in augmented_variants:
            processed_data.append(variant)
            labels.append(label)

    # ---------------------------------------------------------------- #
    #  6. Simpan ke Numpy
    # ---------------------------------------------------------------- #
    X = np.array(processed_data)
    y = np.array(labels)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)

    print(f"\n[OK] Preprocessing Selesai!")
    print(f"   Shape X : {X.shape}  ->  [Sampel, Timesteps={sequence_length}, Fitur={TOTAL_FEATURES}]")
    print(f"   Shape y : {y.shape}")
    print(f"   Fokus   : {(y == 1).sum()} sampel")
    print(f"   Tdk Fok : {(y == 0).sum()} sampel")
    print(f"   Disimpan ke: {output_dir}")
    print(f"   Label map: {label_map}")


if __name__ == "__main__":
    BASE_DIR      = Path(__file__).resolve().parent.parent.parent
    INPUT_CSV     = BASE_DIR / "data" / "extractions" / "dataset_full.csv"
    OUTPUT_FOLDER = BASE_DIR / "data" / "processed"

    if not INPUT_CSV.exists():
        print(f"Error: File {INPUT_CSV} tidak ditemukan!")
    else:
        preprocess_pose_csv(
            input_csv=str(INPUT_CSV),
            output_dir=str(OUTPUT_FOLDER),
            sequence_length=SEQUENCE_LENGTH,
            min_frames=5
        )
