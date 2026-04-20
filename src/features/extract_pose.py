import cv2
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json

def extract_pose_landmarks(dataset_path, output_csv, stride=1):
    dataset_dir = Path(dataset_path)
    video_files = list(dataset_dir.rglob('*.mp4'))

    data = []
    print(f"Ditemukan {len(video_files)} file video.", flush=True)

    # Initialize pose landmarker menggunakan MediaPipe Tasks API
    print("Initialize MediaPipe Pose Solutions...", flush=True)
    try:
        base_dir = Path(__file__).resolve().parent.parent.parent
        model_path = str(base_dir / "models" / "pose_landmarker_lite.task")
        if not os.path.exists(model_path):
            print(f"Error: Model {model_path} tidak ditemukan!")
            return None

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        print("Pose landmarker berhasil diinisialisasi.", flush=True)
    except Exception as e:
        print(f"Gagal inisialisasi model: {e}")
        return None
    for video_path in tqdm(video_files, desc="Memproses Video"):
        # Tentukan label
        parts = video_path.parts

        main_label = "Unknown"
        if "fokus" in parts:
            main_label = "fokus"
        elif "tidak_fokus" in parts:
            main_label = "tidak_fokus"

        sub_label = video_path.parent.name
        video_id = video_path.name

        cap = cv2.VideoCapture(str(video_path))
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frame jika stride > 1
            if frame_num % stride != 0:
                frame_num += 1
                continue
            
            # Frame sudah dalam format RGB dari preprocessing
            image_rgb = frame

            # Convert OpenCV frame to MediaPipe Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Hit landmarks menggunakan mode tasks 
            results = pose_landmarker.detect(mp_image)

            # Baris data dasar
            row = [video_id, frame_num, main_label, sub_label]

            if results.pose_landmarks:
                # Ambil hanya 17 landmark pertama (0-16)
                for landmark in results.pose_landmarks[0][:17]:  
                    row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            else:
                # Zero-padding untuk 17 landmark
                row.extend([0.0] * (17 * 4))

            data.append(row)

        cap.release()

    # Membuat daftar kolom untuk 17 landmark
    columns = ['video_id', 'frame_num', 'main_label', 'sub_label']
    for i in range(17):
        columns.extend([f'x_{i}', f'y_{i}', f'z_{i}', f'v_{i}'])

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"\nProses selesai. Data disimpan ke {output_csv}", flush=True)
    return df

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python extract_pose.py <dataset_path> <output_csv> [stride]", flush=True)
        sys.exit(1)

    dataset_path = sys.argv[1]
    output_csv = sys.argv[2]
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    extract_pose_landmarks(dataset_path, output_csv, stride=stride)