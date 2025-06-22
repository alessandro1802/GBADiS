import os
import warnings
from glob import glob

from tqdm import tqdm
import cv2
import pandas as pd

from visualize import draw_bboxes, draw_keypoints
from src.detector import YOLOv5
from src.estimator import HRNet, RAFT
from src.tracker import OpticalFlowTracker


warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":
    # UCSD Pedestrian
    input_dir = "./data/UCSD_Anomaly_Dataset.v1p2/"
    subsets = {"UCSDped1": (["Train", "Test"], ".tif"),
               "UCSDped2": (["Train", "Test"], ".tif")}
    sequence_dirname_pattern = "**[0-9]"
    frame_idx_offset = 1
    output_dir = "./data/processed/UCSD_Anomaly_Dataset.v1p2/"

    # # ShanghaiTech Campus
    # input_dir = "./data/shanghaitech/"
    # subsets = {"training": (["videos"], ".avi"),
    #            "testing": (["frames"], ".jpg")}
    # sequence_dirname_pattern = "*"
    # frame_idx_offset = 0
    # output_dir = "./data/processed/shanghaitech/"

    DEVICE = "mps"
    IMAGE_SIZE = (960, 544)
    display = False
    write_results = True

    # Load models
    detector = YOLOv5(model_path="./src/yolov5/weights/yolov5x.pt", device=DEVICE)
    pose_estimator = HRNet(cfg_path="./src/hrnet/experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml",
                           model_path="./src/hrnet/weights/pose_hrnet_w48_384x288.pth", device=DEVICE)
    flow_estimator = RAFT(model_path='./src/raft/weights/raft-sintel.pth', device=DEVICE)

    for subset, (splits, file_extension) in subsets.items():
        subset_dir = os.path.join(input_dir, subset)
        for split in splits:
            subset_path = os.path.join(subset_dir, split)
            for video_path in tqdm(glob(os.path.join(subset_path, sequence_dirname_pattern)),
                                   desc=f"Processing {subset}/{split}"):
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
                    frames = [cv2.resize(cv2.imread(os.path.join(video_path, image_file)),
                                         IMAGE_SIZE) for image_file in image_files]

                video_tracks = []
                video_keypoints = []
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
                    # Update tracker with detections and flow
                    tracks = tracker.update(bboxes, flow)
                    video_tracks.append([[trackID, box.tolist()] for trackID, box in tracks] if tracks else [])
                    # Estimate poses for each tracked person
                    keypoints_list = pose_estimator.infer(frame, [box for _, box in tracks])
                    video_keypoints.append([k.tolist() for k in keypoints_list])
                    if display:
                        # Draw optical flow, bboxes and key-points on the frame
                        img = frame.copy()
                        img = draw_bboxes(img, tracks)
                        for keypoints in keypoints_list:
                            img = draw_keypoints(img, keypoints)
                        # Show results
                        cv2.imshow('Pose Estimation', img)
                        # Break on pressing 'q' or space
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord(' ') or key == ord('q'):
                            break
                if display:
                    cv2.destroyAllWindows()
                if write_results:
                    # Export bboxes and pose estimations
                    video_output_dir = os.path.join(output_dir, subset, split)
                    os.makedirs(video_output_dir, exist_ok=True)
                    video_output_path = os.path.join(video_output_dir, video_path.split('/')[-1] + ".csv")
                    video_tracks_df = pd.DataFrame()
                    for frame_idx, tracks in enumerate(video_tracks):
                        df = pd.DataFrame(tracks, columns=["trackID", "bbox"])
                        df["keypoints"] = video_keypoints[frame_idx]
                        df.insert(0, "frame_idx", frame_idx + frame_idx_offset)
                        video_tracks_df = pd.concat([video_tracks_df, df], axis=0, ignore_index=True)
                    video_tracks_df.to_csv(video_output_path, index=False)
