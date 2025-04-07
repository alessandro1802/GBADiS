import os

import cv2

from src.detector import YOLOv5x
from src.estimator import HRNet


def draw_keypoints(frame, keypoints, threshold=0.3):
    for x, y, conf in keypoints:
        if conf > threshold:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
    return frame


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
        # Draw keypoints on the frame
        for keypoints in keypoints_list:
            frame = draw_keypoints(frame, keypoints)
        # Show results
        cv2.imshow('Pose Estimation', frame)
        # Break on pressing 'q' or space
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') or key == ord('q'):
            break
    cv2.destroyAllWindows()
