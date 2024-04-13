from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8m.pt")

    model_save_dir = "/mnt/d/vault/devhub/object_detection_with_yolo8/models/"
    model_save_name = "apple"

    results = model.train(data="config.yaml", epochs=5, imgsz=640, project=model_save_dir, name=model_save_name)