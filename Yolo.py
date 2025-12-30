from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # model nano (ringan)
    results = model("gallery-15.jpg")  # deteksi
    results[0].show()  # tampilkan hasil

if __name__ == "__main__":
    main()
