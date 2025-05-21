from typing import List, Tuple, Union

import numpy as np
import cv2


def draw_bboxes(image: np.ndarray, tracks: List[List[Union[int, np.ndarray]]],
                font_scale: float = 0.2, color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 1):
    img = image.copy()
    for track_id, bbox in tracks:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, f"ID: {track_id}", (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return img


def draw_keypoints(image: np.ndarray, keypoints: List[np.ndarray],
                   threshold: float = 0.3, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 1):
    img = image.copy()
    for x, y, conf in keypoints:
        if conf > threshold:
            cv2.circle(img, (int(x), int(y)), thickness, color, -1)
    return img


def draw_flow_overlay(image: np.ndarray, flow: np.ndarray):
    hsv = np.zeros_like(image)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    blended = cv2.addWeighted(image, 0.6, flow_rgb, 0.4, 0)
    return blended
