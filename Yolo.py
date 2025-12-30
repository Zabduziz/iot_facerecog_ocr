from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # model nano (ringan)
    results = model("gallery-15.jpg")  # deteksi
    results[0].show()  # tampilkan hasil

def realtime():
    model = YOLO("yolov8n.pt")
    model(source=0, show=True)

if __name__ == "__main__":
    main()
