import numpy as np


class OpticalFlowTracker:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age  # Maximum frames to keep a track alive without matching
        self.min_hits = min_hits  # Minimum detections before track is confirmed
        self.iou_threshold = iou_threshold
        self.tracks = []  # List to store active tracks
        self.next_id = 0  # ID counter for new tracks

    def update(self, boxes, flow=None):
        # Each track is [id, box, age, hit_streak, time_since_update]
        # If no boxes, just update existing tracks
        if len(boxes) == 0:
            for track in self.tracks:
                track[3] = 0  # Reset hit streak
                track[4] += 1  # Increment time since update
            # Remove old tracks
            self.tracks = [t for t in self.tracks if t[4] < self.max_age]
            return [t[0:2] for t in self.tracks if t[3] >= self.min_hits]
        # If no existing tracks, create new ones
        if len(self.tracks) == 0:
            for box in boxes:
                self.tracks.append([self.next_id, box, 1, 1, 0])
                self.next_id += 1
            return [t[0:2] for t in self.tracks if t[3] >= self.min_hits]
        # Predict new locations using flow if available
        if flow is not None:
            H, W = flow.shape[:2]
            for track in self.tracks:
                box = track[1]
                # Skip boxes with any NaN
                if any(np.isnan(box)):
                    continue
                # Use optical flow to predict new box position
                # Convert to integer coordinates and clamp to image bounds
                x1 = max(0, min(W, int(box[0])))
                y1 = max(0, min(H, int(box[1])))
                x2 = max(0, min(W, int(box[2])))
                y2 = max(0, min(H, int(box[3])))
                # Get average flow in the bounding box region
                if x1 < x2 and y1 < y2:  # Valid box
                    region = flow[y1:y2, x1:x2]
                    if region.size > 0:
                        flow_x_avg = np.mean(region[:, :, 0])
                        flow_y_avg = np.mean(region[:, :, 1])
                        # Update box position based on flow
                        track[1] = [
                            box[0] + flow_x_avg,
                            box[1] + flow_y_avg,
                            box[2] + flow_x_avg,
                            box[3] + flow_y_avg
                        ]
        # Match detections to tracks using IoU
        matched_tracks, unmatched_tracks, unmatched_detections = self._match_detections_to_tracks(boxes)
        # Update matched tracks
        for track_idx, detection_idx in matched_tracks:
            self.tracks[track_idx][1] = boxes[detection_idx]  # Update position
            self.tracks[track_idx][2] += 1  # Increment age
            self.tracks[track_idx][3] += 1  # Increment hit streak
            self.tracks[track_idx][4] = 0  # Reset time since update
        # Update unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx][3] = 0  # Reset hit streak
            self.tracks[track_idx][4] += 1  # Increment time since update
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            self.tracks.append([self.next_id, boxes[detection_idx], 1, 1, 0])
            self.next_id += 1
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t[4] < self.max_age]
        # Return active tracks
        return [t[0:2] for t in self.tracks if t[3] >= self.min_hits]

    def _match_detections_to_tracks(self, detections):
        if len(self.tracks) == 0 or len(detections) == 0:
            return [], list(range(len(self.tracks))), list(range(len(detections)))
        # Calculate IoU between each detection and track
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for t, track in enumerate(self.tracks):
            for d, detection in enumerate(detections):
                iou_matrix[t, d] = self._calculate_iou(track[1], detection)
        # Use greedy matching (could be replaced with Hungarian algorithm)
        matched_indices = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))
        # Sort IoU in descending order
        indices = np.argsort(-iou_matrix.flatten())
        indices = [(idx // iou_matrix.shape[1], idx % iou_matrix.shape[1]) for idx in indices]

        used_tracks = set()
        used_detections = set()

        for track_idx, detection_idx in indices:
            # Skip if already matched or below threshold
            if track_idx in used_tracks or detection_idx in used_detections:
                continue
            if iou_matrix[track_idx, detection_idx] < self.iou_threshold:
                continue
            matched_indices.append((track_idx, detection_idx))
            used_tracks.add(track_idx)
            used_detections.add(detection_idx)
            if track_idx in unmatched_tracks:
                unmatched_tracks.remove(track_idx)
            if detection_idx in unmatched_detections:
                unmatched_detections.remove(detection_idx)
        return matched_indices, unmatched_tracks, unmatched_detections

    def _calculate_iou(self, box1, box2):
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        # Calculate area of each box
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        # Check if boxes overlap
        if x1_i >= x2_i or y1_i >= y2_i:
            return 0.0
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
