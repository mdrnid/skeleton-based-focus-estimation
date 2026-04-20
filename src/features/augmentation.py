import numpy as np

# Jumlah landmark × fitur per landmark (x, y, z, v) = 68
# Ditambah 2 engineered features = 70 total per timestep
N_LANDMARKS    = 17
RAW_FEATURES   = N_LANDMARKS * 4   # 68
ENG_FEATURES   = 2                 # ear_ratio, nose_offset_x
TOTAL_FEATURES = RAW_FEATURES + ENG_FEATURES  # 70


def apply_jitter(X, sigma=0.005):
    """
    Menambahkan random noise Gaussian pada koordinat landmark.
    Input shape: (timesteps, TOTAL_FEATURES) → (20, 70)
    """
    noise = np.random.normal(0, sigma, X.shape)
    return X + noise


def apply_scaling(X, scale_range=(0.95, 1.05)):
    """
    Melakukan scaling (zoom in/out) pada koordinat x, y, z.
    Hanya mem-scale kolom koordinat spasial, bukan visibility atau engineered features.
    Input shape: (timesteps, TOTAL_FEATURES) → (20, 70)
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])

    # Pisahkan raw landmark part dan engineered features
    X_raw = X[:, :RAW_FEATURES].reshape(X.shape[0], N_LANDMARKS, 4)
    X_eng = X[:, RAW_FEATURES:]   # (timesteps, 2)

    # Scale hanya koordinat x, y, z (indeks 0, 1, 2) — bukan visibility (indeks 3)
    X_raw[:, :, :3] *= scale

    # Rebuild
    X_raw_flat = X_raw.reshape(X.shape[0], RAW_FEATURES)
    return np.concatenate([X_raw_flat, X_eng], axis=1)


def apply_horizontal_flip(X):
    """
    Membalikkan data secara horizontal (mirroring).

    FIX dari versi sebelumnya:
      - Versi lama pakai x_new = 1.0 - x_old  → SALAH jika koordinat sudah
        dinormalisasi (bisa negatif, bukan dalam range [0,1]).
      - Versi baru pakai x_new = -x_old  → BENAR secara geometris setelah
        normalisasi Mid-Shoulder centered.

    Input shape: (timesteps, TOTAL_FEATURES) → (20, 70)
    """
    # Pisahkan raw landmark part dan engineered features
    X_raw = X[:, :RAW_FEATURES].reshape(X.shape[0], N_LANDMARKS, 4).copy()
    X_eng = X[:, RAW_FEATURES:].copy()   # (timesteps, 2)

    # 1. Balik koordinat X dengan negasi (bukan 1 - x)
    X_raw[:, :, 0] = -X_raw[:, :, 0]

    # 2. Tukar indeks landmark kiri & kanan secara anatomis
    #    (dalam 17 landmark pertama MediaPipe Pose)
    left_rights = [
        (1, 4),   # mata kiri ↔ mata kanan
        (2, 5),   # inner eye kiri ↔ kanan
        (3, 6),   # telinga kiri ↔ kanan
        (7, 8),   # mulut kiri ↔ kanan
        (9, 10),  # bahu arah kiri ↔ kanan (inner)
        (11, 12), # bahu kiri ↔ kanan
        (13, 14), # siku kiri ↔ kanan
        (15, 16), # pergelangan kiri ↔ kanan
    ]

    X_flipped = X_raw.copy()
    for left, right in left_rights:
        X_flipped[:, left, :]  = X_raw[:, right, :]
        X_flipped[:, right, :] = X_raw[:, left, :]

    # 3. Update engineered features setelah flip
    #    ear_ratio setelah flip → 1 / ear_ratio (kiri & kanan tertukar)
    #    nose_offset_x setelah flip → -nose_offset_x
    X_eng_flipped = X_eng.copy()
    # ear_ratio (indeks 0)
    ear_ratio_orig = X_eng[:, 0]
    X_eng_flipped[:, 0] = np.where(ear_ratio_orig > 1e-6,
                                   1.0 / ear_ratio_orig,
                                   1.0)
    # nose_offset_x (indeks 1)
    X_eng_flipped[:, 1] = -X_eng[:, 1]

    # Rebuild
    X_raw_flat = X_flipped.reshape(X.shape[0], RAW_FEATURES)
    return np.concatenate([X_raw_flat, X_eng_flipped], axis=1)
