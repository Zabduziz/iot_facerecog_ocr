import cv2
import numpy as np
import time
from picamera2 import Picamera2
from insightface.app import FaceAnalysis

# ================================
# INIT MODEL INSIGHTFACE
# ================================
app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# ================================
# FUNGSI UNTUK MENGAMBIL EMBEDDING
# ================================
def get_embedding(image):
    faces = app.get(image)
    if len(faces) == 0:
        raise ValueError("Tidak ada wajah terdeteksi!")
    
    # Ambil wajah pertama
    face = faces[0]
    return face.normed_embedding

# ================================
# INISIALISASI PICAMERA2
# ================================
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())

# ================================
# FUNGSI UNTUK MENGAMBIL GAMBAR DARI PICAMERA2
# ================================
def capture_image():
    picam2.start()
    time.sleep(1)  # Beri waktu untuk kamera menyesuaikan
    image = picam2.capture_array()
    picam2.stop()
    return image

# ================================
# SCAN WAJAH PERTAMA
# ================================
print("Silakan tampilkan wajah pertama dalam 3 detik...")
time.sleep(3)
img1 = capture_image()
print("Wajah pertama dipindai!")

# ================================
# SCAN WAJAH KEDUA
# ================================
print("Silakan tampilkan wajah kedua dalam 3 detik...")
time.sleep(3)
img2 = capture_image()
print("Wajah kedua dipindai!")

# ================================
# AMBIL EMBEDDING DARI KEDUA GAMBAR
# ================================
emb1 = get_embedding(img1)
emb2 = get_embedding(img2)

# ================================
# HITUNG COSINE SIMILARITY
# ================================
similarity = np.dot(emb1, emb2)

print("Similarity Score:", similarity)

# Threshold umum ArcFace: 0.3 - 0.5
threshold = 0.3

if similarity > threshold:
    print("✓ Wajah Sama / Match")
else:
    print("✗ Wajah Berbeda / Not Match")
