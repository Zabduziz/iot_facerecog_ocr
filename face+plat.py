from ultralytics import YOLO
from insightface.app import FaceAnalysis
from fast_plate_ocr import LicensePlateRecognizer
import cv2
import os
import numpy as np

m = LicensePlateRecognizer('cct-xs-v1-global-model')
yolo_model = YOLO("best.pt")
app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=0)

def face_recog(face_img):
    face = app.get(face_img)
    if len(face) == 0:
        raise ValueError("Tidak terdeteksi wajah")
    face = face[0]
    return face.normed_embedding


def ocr_plate(plate_img):
    #img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    alpr_result = m.run(plate_img)
    for r in alpr_result:
        plate_number = r
    return plate_number


def getInfoUsingModel(img_path):
    img = cv2.imread(img_path)
    results = yolo_model(img, conf=0.4)
    os.makedirs("crops", exist_ok=True)
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        filename = f"crops/obj_{i}_cls{cls_id}_{conf:.2f}.jpg"
        cv2.imwrite(filename, crop)
        
        if cls_id == 0:  # Wajah
            identity = face_recog(crop)
            
        elif cls_id == 1:  # Plat nomor
            plate_text = ocr_plate(crop)

    return identity, plate_text

img_path_1 = "./photo/zaidan1.jpeg"
img_path_2 = "./photo/fajar2.jpeg"

face_result_1, plate_result_1 = getInfoUsingModel(img_path_1)
print("----------- Scan Pertama --------------")
print("Identity:", face_result_1)
print("Plat Nomor:", plate_result_1)
face_result_2, plate_result_2 = getInfoUsingModel(img_path_2)
print("----------- Scan Kedua --------------")
print("Identity:", face_result_2)
print("Plat Nomor:", plate_result_2)

print("----------- Hasil --------------")
similarity = np.dot(face_result_1, face_result_2)
print("Similarity:", similarity)

if similarity > 0.4:
    print("Identitas sama")
else:
    print("Identitas berbeda")
