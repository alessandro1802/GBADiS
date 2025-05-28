import os
import warnings
import json
from glob import glob

from tqdm import tqdm
import cv2

from visualize import draw_bboxes, draw_keypoints
from src.detector import YOLOv5x
from src.estimator import HRNet, RAFT
from src.tracker import OpticalFlowTracker


warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":
    # UCSD Pedestrian
    # input_dir = "./data/UCSD_Anomaly_Dataset.v1p2/"
    # subsets = {"UCSDped1": (["Train", "Test"], ".tif"),
    #            "UCSDped2": (["Test"], ".tif")}
    # sequence_dirname_pattern = f"{split}**[0-9]"
    # frame_idx_offset = 1
    # output_dir = "./data/processed/UCSD_Anomaly_Dataset.v1p2/"

    # ShanghaiTech Campus
    input_dir = "./data/shanghaitech/"
    subsets = {"training": (["videos"], ".avi"),
               "testing": (["frames"], ".jpg")}
    sequence_dirname_pattern = "*"
    frame_idx_offset = 0
    output_dir = "./data/processed/shanghaitech/"

    DEVICE = "mps"
    display = False
    write_results = True

    # Load models
    detector = YOLOv5x(model_path="./src/yolov5/weights/yolov5x.pt", device=DEVICE)
    pose_estimator = HRNet(cfg_path="./src/hrnet/experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml",
                           model_path="./src/hrnet/weights/pose_hrnet_w48_384x288.pth", device=DEVICE)
    flow_estimator = RAFT(model_path='./src/raft/weights/raft-sintel.pth', device=DEVICE)

    for subset, (splits, file_extension) in subsets.items():
        subset_dir = os.path.join(input_dir, subset)
        for split in splits:
            subset_path = os.path.join(subset_dir, split)
            for video_path in tqdm(glob(os.path.join(subset_path, sequence_dirname_pattern)), desc=f"Processing {subset}/{split}"):
                # Load video frames
                if file_extension == ".avi":
                    frames = []
                    cap = cv2.VideoCapture(video_path)
                    success = True
                    while success:
                        success, frame = cap.read()
                        if frame is not None:
                            frames.append(frame)
                else:
                    image_files = sorted([f for f in os.listdir(video_path) if f.endswith(file_extension)])
                    if not image_files:
                        raise Exception(f"No {file_extension} files found in the directory.")
                    frames = [cv2.imread(os.path.join(video_path, image_file)) for image_file in image_files]

                if write_results:
                    # Create output directories
                    output_path = os.path.join(output_dir, subset, split, video_path.split('/')[-1])
                    output_path_json = os.path.join(output_path, "json")
                    os.makedirs(output_path_json, exist_ok=True)
                # Process video frames
                tracker = OpticalFlowTracker(max_age=20, min_hits=3, iou_threshold=0.3)
                prev_frame = None
                for frame_idx, frame in enumerate(frames):
                    # Estimate optical flow
                    if prev_frame is not None:
                        flow = flow_estimator.compute_flow(prev_frame, frame)
                    else:
                        flow = None
                    prev_frame = frame.copy()
                    # Detect people in the frame
                    bboxes = detector.detect(frame)
                    # Estimate poses for each detected person
                    keypoints_list = pose_estimator.infer(frame, bboxes)
                    # Update tracker with detections and flow
                    tracks = tracker.update(bboxes, flow)
                    if write_results:
                        # Export bboxes and pose estimations
                        with open(os.path.join(output_path_json, f"{frame_idx + frame_idx_offset:03d}.json"), 'w') as f:
                            json.dump({
                                # (N, 5)
                                "tracks": [[trackID, box.tolist()] for trackID, box in tracks] if tracks else [],
                                # (N, num_joints, n_features) = (N, 17, 3))
                                "keypoints": [k.tolist() for k in keypoints_list],
                            }, f)
                    if display:
                        # Draw optical flow, bboxes and key-points on the frame
                        img = frame.copy()
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
