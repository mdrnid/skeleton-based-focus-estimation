import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def resample_sequence(frames_features, target_length):
    """
    Melakukan interpolasi temporal (linear) untuk mengubah panjang sekuens
    dari N frame menjadi tepat target_length frame.
    """
    n_frames, n_features = frames_features.shape
    if n_frames == target_length:
        return frames_features
    
    # Buat indeks waktu asal dan target
    src_idx = np.arange(n_frames)
    target_idx = np.linspace(0, n_frames - 1, target_length)
    
    # Interpolasi per fitur
    resampled = np.zeros((target_length, n_features))
    for i in range(n_features):
        resampled[:, i] = np.interp(target_idx, src_idx, frames_features[:, i])
        
    return resampled

def preprocess_pose_csv(input_csv, output_dir, sequence_length=10, min_frames=3):
    """
    Preprocessing data CSV landmark untuk siap masuk ke model LSTM.
    Mencakup: Normalisasi Spasial, Interpolasi Temporal (Resampling), dan Labeling.
    """
    print(f"Membaca data dari {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # 1. Label Mapping Manual (Menjamin konsistensi 1=fokus, 0=tidak_fokus)
    print("Mapping labels...")
    label_map = {"fokus": 1, "tidak_fokus": 0}
    df['label_idx'] = df['main_label'].map(label_map).fillna(-1).astype(int)
    
    # Buang data yang labelnya tidak dikenal
    df = df[df['label_idx'] != -1]

    # Identifikasi kolom koordinat (x_0, y_0, z_0, v_0, ..., v_16)
    landmark_cols = []
    for i in range(17):
        landmark_cols.extend([f'x_{i}', f'y_{i}', f'z_{i}', f'v_{i}'])

    processed_data = []
    labels = []
    
    video_groups = df.groupby('video_id')
    print(f"Memproses {len(video_groups)} video...")

    for vid_id, group in video_groups:
        # Urutkan berdasarkan frame_num
        group = group.sort_values('frame_num')
        
        # Ambil label
        label = group['label_idx'].iloc[0]
        
        # Filter: Buang video yang terlalu pendek
        if len(group) < min_frames:
            continue
            
        # 2. Normalisasi Spasial (Per Frame)
        frames_features = []
        for _, row in group.iterrows():
            coords = row[landmark_cols].values.astype(np.float32)
            
            # Reshape ke (17, 4) -> [x, y, z, v]
            coords_reshaped = coords.reshape(17, 4)
            
            # --- CENTERING (Relatif terhadap Hidung / Landmark 0) ---
            nose_x, nose_y, nose_z = coords_reshaped[0, 0:3]
            coords_reshaped[:, 0] -= nose_x
            coords_reshaped[:, 1] -= nose_y
            coords_reshaped[:, 2] -= nose_z
            
            # --- SCALING (Berdasarkan lebar bahu / Landmark 11 & 12) ---
            l_shoulder = coords_reshaped[11, 0:3]
            r_shoulder = coords_reshaped[12, 0:3]
            shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)
            
            if shoulder_width > 1e-6:
                coords_reshaped[:, 0:3] /= shoulder_width
            
            frames_features.append(coords_reshaped.flatten())
            
        frames_features = np.array(frames_features)
        
        # 3. NORMALISASI TEMPORAL (INTERPOLASI)
        # Mengubah durasi video apa pun menjadi tepat sequence_length frame
        frames_resampled = resample_sequence(frames_features, sequence_length)
            
        processed_data.append(frames_resampled)
        labels.append(label)

    # 4. Save to Numpy
    X = np.array(processed_data)
    y = np.array(labels)
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    
    print(f"\nPreprocessing Selesai!")
    print(f"Shape Fitur (X): {X.shape} -> [Sampel, Timesteps, Fitur]")
    print(f"Shape Label (y): {y.shape}")
    print(f"Data disimpan ke folder: {output_dir}")
    print(f"Mapping: {label_map}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    INPUT_CSV = BASE_DIR / "data" / "extractions" / "dataset_full.csv"
    OUTPUT_FOLDER = BASE_DIR / "data" / "processed"
    
    if not INPUT_CSV.exists():
        print(f"Error: File {INPUT_CSV} tidak ditemukan!")
    else:
        preprocess_pose_csv(
            input_csv=str(INPUT_CSV),
            output_dir=str(OUTPUT_FOLDER),
            sequence_length=10, 
            min_frames=3       
        )
