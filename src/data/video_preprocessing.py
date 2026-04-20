import cv2
import numpy as np
import os
import random
from pathlib import Path

def get_crop_params(width, height):
    """
    Menentukan parameter cropping untuk menghilangkan area statis di latar belakang.
    Bisa disesuaikan dengan environment kelas/kamera.
    Misalnya: membuang 10% margin di kiri, kanan, dan atas.
    """
    margin_x = int(width * 0.10)
    margin_y = int(height * 0.10)
    
    x1 = margin_x
    y1 = margin_y
    x2 = width - margin_x
    y2 = height # Keep the bottom where the body/desk usually is
    return x1, y1, x2, y2

def apply_clahe(frame):
    """
    Normalisasi Pencahayaan Ekstrim menggunakan CLAHE.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Create and apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def add_gaussian_noise(image):
    """
    Robustness Injection: Menambahkan Gaussian Noise
    """
    row, col, ch = image.shape
    mean = 0
    var = 15  # Varian noise
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def adjust_brightness_jitter(image):
    """
    Robustness Injection: Fluktuasi cahaya (Brightness Jitter)
    """
    value = random.randint(-40, 40)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if value >= 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        lim = 0 - value
        v[v < lim] = 0
        v[v >= lim] -= abs(value)

    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image

def preprocess_video(input_path, output_path, target_fps=15, target_height=480, is_training=False):
    """
    Mengeksekusi tahapan preprocessing (1-5) pada satu file video.
    """
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error membuka video: {input_path}")
        return False

    # 1. Parameter Temporal Subsampling
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 30 # Default assumption
    
    frame_skip = max(1, int(round(orig_fps / target_fps)))
    out_fps = orig_fps / frame_skip
    
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if orig_height == 0 or orig_width == 0:
        return False

    # Dapatkan parameter potongan statis
    x1, y1, x2, y2 = get_crop_params(orig_width, orig_height)
    crop_w = x2 - x1
    crop_h = y2 - y1

    # 2. Parameter Frame Resizing
    aspect_ratio = crop_w / crop_h
    new_height = target_height
    new_width = int(new_height * aspect_ratio)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (new_width, new_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Temporal Subsampling
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        # 4. Static ROI Cropping
        frame_cropped = frame[y1:y2, x1:x2]

        # 2. Frame Resizing
        frame_resized = cv2.resize(frame_cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 3. CLAHE Normalisasi Pencahayaan
        frame_processed = apply_clahe(frame_resized)

        # 5. Robustness Injection (Khusus Training)
        if is_training:
            # Aplikasikan secara acak untuk mensimulasikan lingkungan yang dinamis
            if random.random() < 0.3: # probabilitas 30% pada frame tersebut
                frame_processed = add_gaussian_noise(frame_processed)
            if random.random() < 0.3: # probabilitas 30% fluktuasi
                frame_processed = adjust_brightness_jitter(frame_processed)

        # 6. Konversi Warna (Opsional ke RGB sebelum disave jika diperlukan algoritma spesifik)
        # Catatan: cv2.VideoWriter secara default mengharapkan BGR. Jika kita paksa Save RGB, 
        # maka video yang diputar di memutar biasa akan terbalik warnanya (biru/merah tertukar).
        # Tapi demi instruksi pipe kita convert untuk melihat efeknya.
        # Atau cukup di-convert di tahap ekstraksi saja. 
        # Di sini kita letakkan opsional konversi BGR ke RGB untuk visualisasi raw RGB jika MP membutuhkannya.
        frame_final = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB) 

        # Write frame (Jika memakai cv2 VideoWriter standar, akan menyimpannya sebagai channel 1,2,3 - RGB diterjemahkan sbg BGR)
        # Jika ingin menyimpan warna yg benar di OpenCV output, kembalikan ke BGR, namun jika instruksinya adalah 
        # murni convert, kita simpan frame_final.
        out.write(frame_final)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Selesai: {os.path.basename(input_path)} -> {new_width}x{new_height} @ {out_fps:.1f} FPS")
    return True

def process_dataset(input_dir, output_dir, target_fps=15, target_height=480, is_training=False):
    """
    Meliterate semua folder di dataset dan menjalankan preprocessing.
    Dengan dukungan pencarian bersarang (recursive) untuk struktur folder berlapis.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Buat direktori output jika belum ada
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Cari semua file mp4 bahkan di dalam sub-folder yang dalam
    video_files = list(input_path.rglob('*.mp4'))
    
    if not video_files:
        print(f"Peringatan: Tidak ditemukan file .mp4 di dalam directory {input_path}")
        return
        
    for in_file in video_files:
        # Lewati folder output misal namanya dataset_preprocessed
        if 'preprocessed' in str(in_file):
            continue
            
        # Dapatkan struktur path yang sama (relatif) dari input directory
        # misal fokus/fokus/melihat_layar/1.mp4
        rel_path = in_file.relative_to(input_path)
        
        out_file = output_path / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        preprocess_video(
            str(in_file), 
            str(out_file), 
            target_fps=target_fps, 
            target_height=target_height, 
            is_training=is_training
        )

if __name__ == "__main__":
    # Konfigurasi Path setelah restrukturisasi
    # Gunakan absolute/relative path merujuk ke root
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATASET_DIR = str(BASE_DIR / "data" / "raw")
    OUTPUT_DIR = str(BASE_DIR / "data" / "processed")
    
    print(f"Dataset Dir: {DATASET_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print("Memulai Preprocessing Level Video...")
    
    # Menjalankan pipeline: FPS 15, resolusi 640x480.
    process_dataset(
        input_dir=DATASET_DIR, 
        output_dir=OUTPUT_DIR, 
        target_fps=10,           # Mengurangi beban (sebelumnya 15)
        target_height=480,       # Resolution terbaik vs trade-off memori komputer
        is_training=True         # Nyalakan training mode untuk augmentasi jika perlu
    )
    print("Proses Selesai!")
