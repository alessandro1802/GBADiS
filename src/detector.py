import torch
from ultralytics import YOLO


class YOLOv5:
    def __init__(self, model_path="./yolov5/weights/yolov5x.pt", device="cpu"):
        # Load the YOLOv5 model with the local weights
        self.device = device
        self.model = torch.hub.load("ultralytics/yolov5", "custom", model_path)
        self.model.to(self.device)

    def detect(self, frame):
        # Process the frame
        results = self.model(frame)
        detections = results.xyxyn[0].cpu().numpy()
        # Get person detections (class 0)
        person_boxes = [det[:4] for det in detections if int(det[5]) == 0]
        return person_boxes


class YOLOv11:
    def __init__(self, model_path="./yolov11/weights/yolo11n-pose.pt",
                 tracker_config_path = "./yolov11/trackers/botsort.yaml",
                 device="cpu"):
        # Load the YOLOv5 model with the local weights
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.tracker_config_path = tracker_config_path

    def detect(self, frame):
        # Process the frame
        results = self.model.track(frame, persist=True, tracker=self.tracker_config_path,
                                   classes=[0], conf=0.01)
        return results


if __name__ == "__main__":
    import cv2
    DEVICE = "mps"
    # image_path = "../data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001/001.tif"
    #
    # detector = YOLOv5(model_path="./yolov5/weights/yolov5x.pt", device=DEVICE)
    # image = cv2.imread(image_path)
    # results = detector.detect(image)

    import os
    IMAGE_SIZE = (960, 544)
    video_path = "../data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001"
    image_files = sorted([f for f in os.listdir(video_path) if f.endswith(".tif")])
    images = [cv2.resize(cv2.imread(os.path.join(video_path, image_path)),
                         IMAGE_SIZE) for image_path in image_files][:15]

    detector = YOLOv11(model_path="./yolov11/weights/yolo11s-pose.pt",
                       tracker_config_path="./yolov11/trackers/botsort.yaml",
                       device=DEVICE)
    results = detector.detect(images)
    for result in results:
        annotated_frame = result.plot()
        # Display the annotated frame
        cv2.imshow("YOLOv11 Tracking", annotated_frame)
        # Press 'q' to stop
        if cv2.waitKey(1000) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
