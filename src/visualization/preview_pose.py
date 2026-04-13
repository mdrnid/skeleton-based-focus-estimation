import cv2
import pandas as pd
import random
from pathlib import Path
import os

def preview_pose_from_csv(csv_path):
    """
    Membaca data landmark dari CSV dan menampilkannya kembali di video asli.
    """
    print(f"Loading data dari: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Error: File CSV tidak ditemukan di {csv_path}")
        return

    # 1. Load CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Gagal membaca CSV: {e}")
        return
    
    if df.empty:
        print("CSV data kosong.")
        return

    # 2. Ambil satu video secara acak atau berdasarkan input
    video_ids = df['video_id'].unique()
    video_id = random.choice(video_ids)
    print(f"\n--- Preview Pose Data ---")
    print(f"Video ID: {video_id}")
    
    # 3. Filter data untuk video tersebut
    video_data = df[df['video_id'] == video_id].sort_values('frame_num')
    
    # 4. Cari lokasi file video tersebut
    # Kita cari di data/processed dan data/raw
    base_dir = Path(csv_path).resolve().parent.parent.parent
    search_dirs = [
        base_dir / "data" / "processed",
        base_dir / "data" / "raw"
    ]
    
    video_path = None
    for s_dir in search_dirs:
        if not s_dir.exists(): continue
        found = list(s_dir.rglob(video_id))
        if found:
            video_path = found[0]
            break
            
    if not video_path:
        print(f"Error: File video '{video_id}' tidak ditemukan di folder data.")
        return

    print(f"File ditemukan: {video_path}")

    # 5. Inisialisasi Video Capture
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30 # Fallback

    # Koneksi pose MediaPipe (Hanya 0-16: Wajah dan Lengan)
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    print("\nMenampilkan Preview. Tekan 'q' untuk berhenti.")

    # Loop utama untuk looping video
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        for _, row in video_data.iterrows():
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            
            # --- Drawing Skeleton ---
            # 1. Gambar Garis (Connections)
            for connection in POSE_CONNECTIONS:
                s_idx, e_idx = connection
                # Cek visibilitas landmark (menggunakan f-string untuk kolom)
                try:
                    if row[f'v_{s_idx}'] > 0.5 and row[f'v_{e_idx}'] > 0.5:
                        pt1 = (int(row[f'x_{s_idx}'] * width), int(row[f'y_{s_idx}'] * height))
                        pt2 = (int(row[f'x_{e_idx}'] * width), int(row[f'y_{e_idx}'] * height))
                        cv2.line(display_frame, pt1, pt2, (255, 255, 255), 2)
                except KeyError:
                    continue
            
            # 2. Gambar Titik (Landmarks 0-16)
            for i in range(17):
                try:
                    if row[f'v_{i}'] > 0.5:
                        cx, cy = int(row[f'x_{i}'] * width), int(row[f'y_{i}'] * height)
                        
                        # Warna berbeda untuk estetika
                        color = (0, 255, 0) # Hijau (Tubuh)
                        if i < 11: color = (255, 0, 0) # Biru (Wajah)
                        elif i in [11, 12, 23, 24]: color = (0, 0, 255) # Merah (Sendi Utama)
                        
                        cv2.circle(display_frame, (cx, cy), 4, color, -1)
                        cv2.circle(display_frame, (cx, cy), 6, (255, 255, 255), 1)
                except KeyError:
                    continue

            # --- Overlay Informasi ---
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 70), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)

            cv2.putText(display_frame, f"Video: {video_id}", (20, 30), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, f"Frame: {int(row['frame_num'])} | Label: {row['main_label']} ({row['sub_label']})", (20, 60), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(display_frame, "Q to Exit", (max(20, width - 150), 35), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow('Pose Extraction Preview', display_frame)
            
            # Kontrol kecepatan frame (sinkronisasi dengan FPS asli)
            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("Preview dihentikan oleh user.")
                return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Inisialisasi Path
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    CSV_PATH = BASE_DIR / "data" / "extractions" / "dataset_full.csv"
    
    preview_pose_from_csv(CSV_PATH)
