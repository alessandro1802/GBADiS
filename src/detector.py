import torch


class YOLOv5x:
    def __init__(self, model_path="./yolov5/weights/yolov5x.pt", device="cpu"):
        # Load the YOLOv5 model with the local weights
        self.device = device
        self.model = torch.hub.load("ultralytics/yolov5", "custom", model_path)
        self.model.to(self.device)

    def detect(self, frame):
        # Process the frame
        results = self.model(frame)
        # Get person detections (class 0)
        detections = results.xyxy[0].cpu().numpy()
        person_boxes = [det[:4] for det in detections if int(det[5]) == 0]
        return person_boxes
