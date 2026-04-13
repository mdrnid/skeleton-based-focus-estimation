import numpy as np

def apply_jitter(X, sigma=0.005):
    """
    Menambahkan random noise pada koordinat landmark.
    Input shape: (timesteps, features) -> (10, 68)
    """
    noise = np.random.normal(0, sigma, X.shape)
    # Jangan tambahkan noise pada kolom visibilitas jika diinginkan, 
    # tapi secara praktis noise kecil pada visibilitas tidak masalah.
    return X + noise

def apply_scaling(X, scale_range=(0.95, 1.05)):
    """
    Melakukan scaling (zoom in/out) pada koordinat x dan y.
    Input shape: (timesteps, features)
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    
    # Reshape ke (timesteps, 17, 4) untuk memisahkan x, y, z, v
    X_reshaped = X.reshape(X.shape[0], 17, 4)
    
    # Scale hanya pada koordinat x dan y (indeks 0 dan 1)
    X_reshaped[:, :, :2] *= scale
    
    return X_reshaped.reshape(X.shape)

def apply_horizontal_flip(X):
    """
    Membalikkan data secara horizontal (mirroring).
    Input shape: (timesteps, features)
    """
    X_reshaped = X.reshape(X.shape[0], 17, 4)
    
    # 1. Balik koordinat x: x_new = 1.0 - x_old
    X_reshaped[:, :, 0] = 1.0 - X_reshaped[:, :, 0]
    
    # 2. Tukar indeks landmark kiri & kanan
    # Pasangan yang harus ditukar (0-16):
    # (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)
    left_rights = [
        (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)
    ]
    
    X_flipped = X_reshaped.copy()
    for left, right in left_rights:
        X_flipped[:, left, :] = X_reshaped[:, right, :]
        X_flipped[:, right, :] = X_reshaped[:, left, :]
        
    return X_flipped.reshape(X.shape)
