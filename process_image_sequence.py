import os
import warnings
import json
from glob import glob
from tqdm import tqdm

import cv2
import numpy as np

from src.detector import YOLOv5x
from src.estimator import HRNet, RAFT
from src.tracker import OpticalFlowTracker


warnings.simplefilter(action='ignore', category=FutureWarning)

# TODO remove
def draw_flow_overlay(image, flow):
    hsv = np.zeros_like(image)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    blended = cv2.addWeighted(image, 0.6, flow_rgb, 0.4, 0)
    return blended


def draw_bboxes(image, tracks, font_scale=0.2, color=(0, 0, 255), thickness=1):
    img = image.copy()
    for track_id, bbox in tracks:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, f"ID: {track_id}", (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
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
    write_results = True

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
                if write_results:
                    # Create output directories
                    output_path = os.path.join(output_dir, subset, split, video_path.split('/')[-1])
                    output_path_json = os.path.join(output_path, "json")
                    ### TODO remove
                    if os.path.exists(output_path_json):
                        continue
                    # output_path_flow = os.path.join(output_path, "flow")
                    # os.makedirs(output_path_flow, exist_ok=True)
                    ###
                    os.makedirs(output_path_json, exist_ok=True)
                # Process video frames
                tracker = OpticalFlowTracker(max_age=20, min_hits=3, iou_threshold=0.3)
                prev_frame = None
                for frame_idx, frame in enumerate(frames):
                    # Estimate optical flow
                    if prev_frame is not None:
                        ### TODO remove
                        exported_flow_path = os.path.join("./data/processed/UCSD_Anomaly_Dataset.v1p2_OLD", subset, split, video_path.split('/')[-1], "flow")
                        if os.path.exists(exported_flow_path):
                            pass
                            flow = np.load(os.path.join(exported_flow_path, f"{frame_idx + 1:03d}.npy"))
                        else:
                            flow = flow_estimator.compute_flow(prev_frame, frame)
                        ###
                        # flow = flow_estimator.compute_flow(prev_frame, frame) # TODO uncomment
                    else:
                        flow = None
                    prev_frame = frame.copy()
                    # Detect people in the frame
                    bboxes = detector.detect(frame)
                    # Estimate poses for each detected person
                    keypoints_list = pose_estimator.infer(frame, bboxes)
                    # Update tracker with detections and flow
                    try: ### TODO remove
                        tracks = tracker.update(bboxes, flow)
                    ### TODO remove
                    except Exception as e:
                        print(frame_idx+1, bboxes)
                        if any([np.isnan(coord) for box in bboxes for coord in box]):
                            bboxes = detector.detect(frame)
                            tracks = tracker.update(bboxes, flow)
                    ###
                    if write_results:
                        # Export bboxes and pose estimations
                        if tracks:
                            with open(os.path.join(output_path_json, f"{frame_idx + 1:03d}.json"), 'w') as f:
                                json.dump({
                                    # (N, 5)
                                    "tracks": [[trackID, box.tolist()] for trackID, box in tracks],
                                    # (N, num_joints, n_features) = (N, 17, 3))
                                    "keypoints": [k.tolist() for k in keypoints_list],
                                }, f)
                        ### TODO remove
                        # # Export optical flow
                        # if flow is not None:
                        #     # (H, W, 3)
                        #     np.save(os.path.join(output_path_flow, f"{frame_idx + 1:03d}.npy"), flow)
                        ###
                    if display:
                        # Draw optical flow, bboxes and key-points on the frame
                        img = frame.copy()
                        ### TODO remove
                        # if flow is not None:
                        #     img = draw_flow_overlay(img, flow)
                        ###
                        img = draw_bboxes(img, tracks)
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
