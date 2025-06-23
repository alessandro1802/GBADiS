from typing import List, Tuple, Union

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def draw_bboxes(image: np.ndarray, tracks: List[List[Union[int, np.ndarray]]],
                font_scale: float = 0.2, color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 1) -> np.ndarray:
    img = image.copy()
    for track_id, bbox in tracks:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, f"ID: {track_id}", (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return img


def draw_keypoints(image: np.ndarray, keypoints: List[np.ndarray],
                   threshold: float = 0.3, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 1) -> np.ndarray:
    img = image.copy()
    for x, y, conf in keypoints:
        if conf > threshold:
            cv2.circle(img, (int(x), int(y)), thickness, color, -1)
    return img


def draw_flow_overlay(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    hsv = np.zeros_like(image)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    blended = cv2.addWeighted(image, 0.6, flow_rgb, 0.4, 0)
    return blended


# Offset = video_len - len(extracted_labels[0])
def plot_anomaly_scores(labels: np.ndarray, scores: np.ndarray, offset: int = 6) -> None:
    fig = plt.figure(figsize=(15, 4))
    # Plot anomaly scores
    plt.plot(np.arange(offset, len(scores) + offset), scores, label='Anomaly Score', color='blue')
    # Highlight abnormal regions
    in_abnormal = False
    start_idx = 0
    for i, val in enumerate(labels):
        if val == 1 and not in_abnormal:
            start_idx = i
            in_abnormal = True
        elif val == 0 and in_abnormal:
            plt.axvspan(start_idx + offset, i + offset, color='red', alpha=0.3)
            in_abnormal = False
    # Handle case where anomaly extends to the end
    if in_abnormal:
        plt.axvspan(start_idx + offset, len(labels) + offset, color='red', alpha=0.3)
    plt.xlabel("Frame")
    plt.ylabel("Anomaly Score")
    plt.title("Anomaly score vs Ground Truth abnormal regions")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    return fig


def plot_roc_curve(true_y: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    return fig, roc_auc
