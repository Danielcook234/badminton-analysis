from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('yolov8n.pt')

    results = model.train(
        data = 'shuttlecock_data.yaml',
        epochs = 50,
        imgsz = 640,
        batch = 16,
        name = "shuttlecock_yolov8n"
    )

    print("training complete")