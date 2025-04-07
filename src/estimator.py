from types import SimpleNamespace

import cv2
import torch
import numpy as np

from src.hrnet.lib.config import cfg
from src.hrnet.lib.config.default import update_config
from src.hrnet.lib.models import pose_hrnet
from src.hrnet.lib.utils.transforms import get_affine_transform, flip_back
from src.hrnet.lib.core.inference import get_final_preds


class HRNet:
    def __init__(self,
                 cfg_path="./hrnet/experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml",
                 model_path="./hrnet/weights/pose_hrnet_w48_384x288.pth",
                 device="cpu"):
        # Load config and update global cfg
        args = SimpleNamespace(
            cfg=cfg_path,
            opts=[],  # no overrides
            modelDir='',
            logDir='',
            dataDir=''  # or path to model/data root if needed
        )
        self.cfg = cfg
        # COCO flip pairs
        self.cfg.DATASET.FLIP_PAIRS = [
            [1, 2], [3, 4], [5, 6], [7, 8],
            [9, 10], [11, 12], [13, 14], [15, 16]
        ]
        update_config(self.cfg, args)
        self.device = device
        # Initialize model
        self.model = pose_hrnet.get_pose_net(cfg, is_train=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        # Image size from config
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.aspect_ratio = self.image_size[0] * 1.0 / self.image_size[1]
        self.input_size = tuple(self.image_size.astype(int))


    def infer(self, image, bboxes):
        """
        Run pose estimation on all bounding boxes in the given image.
        Args:
            image (np.ndarray): BGR image.
            bboxes (List[List[int]]): List of [x1, y1, x2, y2] bounding boxes.
        Returns:
            List[np.ndarray]: List of keypoints arrays (num_joints x 3).
        """
        keypoints_all = []
        for bbox in bboxes:
            input_tensor, center, scale, _ = self.preprocess(image, bbox)
            with torch.no_grad():
                output = self.model(input_tensor)
                if self.cfg.TEST.FLIP_TEST:
                    flipped = torch.flip(input_tensor, [3])
                    flipped_output = self.model(flipped)
                    flipped_output = flip_back(flipped_output.cpu().numpy(), self.cfg.DATASET.FLIP_PAIRS)
                    output = (output.cpu().numpy() + flipped_output) * 0.5
                else:
                    output = output.cpu().numpy()

                preds, confs = get_final_preds(self.cfg, output, np.array([center]), np.array([scale]))
                keypoints_all.append(np.concatenate([preds[0], confs[0]], axis=1))  # shape: (num_joints, 3))
        return keypoints_all

    def preprocess(self, image, bbox):
        """
        Prepare an input tensor for a single person from the image and bbox.
        Args:
            image (np.ndarray): BGR image.
            bbox (List[int]): [x1, y1, x2, y2] bounding box.
        Returns:
            torch.Tensor: Input tensor (1, 3, H, W)
            np.ndarray: Center
            np.ndarray: Scale
            np.ndarray: Affine transform matrix
        """
        x, y, w, h = self._xyxy_to_center_scale(bbox)
        center = np.array([x, y], dtype=np.float32)
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)

        trans = get_affine_transform(center, scale, 0, self.input_size)
        cropped = cv2.warpAffine(image, trans, self.input_size, flags=cv2.INTER_LINEAR)

        img = cropped.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        return img, center, scale, trans

    def _xyxy_to_center_scale(self, bbox):
        """
        Convert a bounding box to center + scale for affine transform.
        Args:
            bbox (List[int]): [x1, y1, x2, y2]
        Returns:
            Tuple[float, float, float, float]: center_x, center_y, width, height
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        if w > self.aspect_ratio * h:
            h = w / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        return center_x, center_y, w, h
