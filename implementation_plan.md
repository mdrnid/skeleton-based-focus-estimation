# Implementation Plan - Preprocessing Data Pose (CSV to LSTM)

Membangun pipeline untuk mengubah data CSV landmark menjadi format array (Numpy) yang siap untuk training model Deep Learning (LSTM/GRU). Script ini akan menangani inkonsistensi jumlah frame antar video dan melakukan normalisasi spasial agar model bersifat *person-invariant*.

## User Review Required

> [!IMPORTANT]
> **Data Placeholder:** File `dataset_opencv_new.csv` saat ini berisi data acak. Script ini akan tetap dibuat agar bisa berjalan pada file tersebut, namun hasil trainingnya tidak akan valid sampai Anda menggunakan data MediaPipe asli.

> [!WARNING]
> **Padding & Truncating:** Video yang memiliki kurang dari 10 frame akan diberikan `Zero Padding` (ditambah baris kosong). Video yang lebih dari 10 frame akan dipotong (truncated) atau dipecah menjadi beberapa sekuens.

## Proposed Changes

### [features]

#### [NEW] [preprocess_csv.py](file:///d:/Arya%20Files/kuliah/UNM_SEM6/COMVIS/Project%20Akhir/src/features/preprocess_csv.py)
Script Python mandiri yang akan melakukan:
1. **Load Data:** Membaca `dataset_opencv_new.csv` menggunakan Pandas.
2. **Normalisasi Relatif (Centering):** 
   - Mengambil koordinat Hidung (Landmark 0) sebagai titik asal (0,0).
   - Mengurangi koordinat semua landmark lain dengan koordinat hidung tersebut.
3. **Penskalaan (Scaling):**
   - Menghitung jarak antar bahu (Landmark 11 & 12).
   - Membagi semua koordinat dengan jarak bahu ini agar model tidak terpengaruh oleh jarak orang ke kamera.
4. **Reshaping & Padding:**
   - Mengelompokkan baris berdasarkan `video_id`.
   - Menggunakan `tf.keras.preprocessing.sequence.pad_sequences` atau Numpy manual untuk memastikan setiap video memiliki tepat 10 frame.
5. **Label Encoding:** 
   - Map `fokus` -> 1, `tidak_fokus` -> 0.
6. **Export:** Menyimpan hasil berupa `X_train.npy` dan `y_train.npy`.

## Open Questions
1. **Filtering:** Apakah video dengan jumlah frame sangat sedikit (misal < 3 frame) sebaiknya **dibuang** saja daripada di-pad? (Sangat disarankan untuk dibuang agar tidak merusak akurasi).
2. **Window Size:** Apakah Anda ingin tetap di angka **10 frame**? (Angka ini cukup pendek, biasanya aksi minimal 15-30 frame / 1 detik).

## Verification Plan

### Automated Tests
- Menjalankan script dan memverifikasi dimensi array output menggunakan perintah:
  `python -c "import numpy as np; data = np.load('data/processed/X.npy'); print(data.shape)"`
  Target: `(N, 10, 132)`.
