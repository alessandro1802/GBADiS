import os
import warnings
import json
from glob import glob
from tqdm import tqdm

import cv2
import numpy as np

from src.detector import YOLOv5x
from src.estimator import HRNet, RAFT


warnings.simplefilter(action='ignore', category=FutureWarning)


def draw_flow_overlay(image, flow):
    hsv = np.zeros_like(image)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    blended = cv2.addWeighted(image, 0.6, flow_rgb, 0.4, 0)
    return blended


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
    input_dir = "./data/UCSD_Anomaly_Dataset.v1p2/"
    subsets = {"UCSDped1": ["Train", "Test"],
               "UCSDped2": ["Test"]}
    output_dir = "./data/processed/UCSD_Anomaly_Dataset.v1p2/"

    DEVICE = "mps"
    display = False
    # Load models
    detector = YOLOv5x(model_path="./src/yolov5/weights/yolov5x.pt", device=DEVICE)
    pose_estimator = HRNet(cfg_path="./src/hrnet/experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml",
                           model_path="./src/hrnet/weights/pose_hrnet_w48_384x288.pth", device=DEVICE)
    flow_estimator = RAFT(model_path='./src/raft/weights/raft-sintel.pth', device=DEVICE)

    for subset, splits in subsets.items():
        subset_dir = os.path.join(input_dir, subset)
        for split in splits:
            subset_path = os.path.join(subset_dir, split)
            for video_path in tqdm(glob(os.path.join(subset_path, f"{split}**[0-9]")), desc=f"Processing {subset}/{split}"):
                # Load video frames
                image_files = sorted([f for f in os.listdir(video_path) if f.endswith('.tif')])
                if not image_files:
                    raise Exception("No TIFF files found in the directory.")
                frames = [cv2.imread(os.path.join(video_path, image_file)) for image_file in image_files]
                # Create output directories
                output_path = os.path.join(output_dir, subset, split, video_path.split('/')[-1])
                output_path_json = os.path.join(output_path, "json")
                os.makedirs(output_path_json, exist_ok=True)
                output_path_flow = os.path.join(output_path, "flow")
                os.makedirs(output_path_flow, exist_ok=True)
                # Process video frames
                prev_frame = None
                for frame_idx, frame in enumerate(frames):
                    # Estimate optical flow
                    if prev_frame is not None:
                        flow = flow_estimator.compute_flow(prev_frame, frame)
                    else:
                        flow = None
                    prev_frame = frame.copy()
                    # Detect people in the frame
                    boxes = detector.detect(frame)
                    # Estimate poses for each detected person
                    keypoints_list = pose_estimator.infer(frame, boxes)
                    # Export bboxes and pose estimations
                    with open(os.path.join(output_path_json, f"{frame_idx + 1:03d}.json"), 'w') as f:
                        json.dump({'boxes': [b.tolist() for b in boxes],  # (N, 4)
                                   'keypoints': [k.tolist() for k in keypoints_list],  # (N, num_joints = (17, 3))
                                   }, f)
                    # Export optical flow
                    if flow is not None:
                        np.save(os.path.join(output_path_flow, f"{frame_idx + 1:03d}.npy"), flow)  # (H, W, 3)
                    # Draw optical flow, bboxes and key-points on the frame
                    if display:
                        img = frame.copy()
                        if flow is not None:
                            img = draw_flow_overlay(img, flow)
                        img = draw_bboxes(img, boxes)
                        for keypoints in keypoints_list:
                            img = draw_keypoints(img, keypoints)
                        # Show results
                        cv2.imshow('Pose Estimation', cv2.resize(img, (960, 540)))
                        # Break on pressing 'q' or space
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord(' ') or key == ord('q'):
                            break
                if display:
                    cv2.destroyAllWindows()
