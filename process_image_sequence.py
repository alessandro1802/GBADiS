import os

import cv2

from src.detector import YOLOv5x
from src.estimator import HRNet


def draw_bboxes(image, boxes, color=(0, 0, 255), thickness=1):
    img = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def draw_keypoints(image, keypoints, threshold=0.3, color=(0, 255, 0), thickness=1):
    img = image.copy()
    for x, y, conf in keypoints:
        if conf > threshold:
            cv2.circle(img, (int(x), int(y)), thickness, color, -1)
    return img


if __name__ == "__main__":
    input_dir = "./data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/Train004/"

    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.tif')])
    if not image_files:
        raise Exception("No TIFF files found in the directory.")

    DEVICE = "mps"
    detector = YOLOv5x(model_path="./src/yolov5/weights/yolov5x.pt", device=DEVICE)
    estimator = HRNet(cfg_path="./src/hrnet/experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml",
                      model_path="./src/hrnet/weights/pose_hrnet_w48_384x288.pth", device=DEVICE)

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        frame = cv2.imread(image_path)
        # Detect people in the frame
        boxes = detector.detect(frame)
        # Estimate poses for each detected person
        keypoints_list = estimator.infer(frame, boxes)
        # Draw bboxes and key-points on the frame
        img = frame.copy()
        img = draw_bboxes(frame, boxes)
        for keypoints in keypoints_list:
            img = draw_keypoints(img, keypoints)
        # Show results
        cv2.imshow('Pose Estimation', img)
        # Break on pressing 'q' or space
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') or key == ord('q'):
            break
    cv2.destroyAllWindows()
