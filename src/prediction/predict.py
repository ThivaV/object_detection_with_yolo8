import os
import cv2
from ultralytics import YOLO

if __name__ == "__main__":
    VIDEOS_DIR = os.path.join("/mnt/d/vault/devhub/object_detection_with_yolo8/data/", "videos")

    extension = ".mp4"
    output_postfix = "_output.mp4"

    # Get a list of all files in the directory
    all_files = os.listdir(VIDEOS_DIR)

    # Filter files with the specified postfix
    matching_output_files = [file for file in all_files if file.lower().endswith(output_postfix.lower())]
    
    # Delete each matching file
    for file in matching_output_files:
        file_path = os.path.join(VIDEOS_DIR, file)
        os.remove(file_path)
        print(f"Removed: {file_path}")
    
    matching_files = [file for file in all_files if file.lower().endswith(extension.lower())]

    for file in matching_files:
        video_path = os.path.join(VIDEOS_DIR, file)
        video_path_out = "{}_output.mp4".format(video_path)

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        H, W, _ = frame.shape
        out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

        model_path = os.path.join("/mnt/d/vault/devhub/object_detection_with_yolo8/models", "apple", "weights", "best.pt")

        # Load a model
        model = YOLO(model_path)

        threshold = 0.5

        while ret:
            results = model(frame)[0]

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result

                if score > threshold:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                    
            out.write(frame)
            ret, frame = cap.read()

        cap.release()
        out.release()
        cv2.destroyAllWindows()