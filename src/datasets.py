import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class SurveillanceAnomalyDataset(Dataset):
    def __init__(self, root_dir: str, num_frames: int = 5, num_nodes: int = 17, max_people: int = 30, transform=None):
        """
        :param root_dir: path to dataset root containing video folders (e.g., Train001).
        :param num_frames: number of consecutive frames per sequence sample (T)
        :param num_nodes: number of joints per person (N, N=17 based on HRNet).
        :param max_people: maximum number of people per video.
        :param transform: optional transform to apply to the data.
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.num_nodes = num_nodes
        self.max_people = max_people
        self.transform = transform
        self.samples = self._gather_samples()
        # Indices of the four torso joints for geometric center calculation.
        self.torso_joint_indices = [4, 5, 10, 11]

    def _gather_samples(self):
        """
        Gathers paths to JSON files for each sequence in the dataset.
        Each sample will contain a sequence of 'num_frames' JSON files.
        """
        samples = []
        for seq_folder in sorted(os.listdir(self.root_dir)):
            seq_path = os.path.join(self.root_dir, seq_folder)
            json_dir = os.path.join(seq_path, 'json')
            if os.path.isdir(json_dir):
                json_files = sorted(glob.glob(os.path.join(json_dir, '*.json')))
                if len(json_files) >= self.num_frames:
                    for i in range(len(json_files) - self.num_frames):
                        samples.append(json_files[i : i + self.num_frames])
        return samples

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Generates one sample of data, consisting of 'num_frames' of skeleton data.
        """
        json_paths = self.samples[idx]
        all_people_keypoints = []  # Store keypoints for all individuals in each frame
        for json_path in json_paths:
            with open(json_path, 'r') as f:
                data = json.load(f)
            keypoint_data = data.get('keypoints', [])
            # boxes = data.get('boxes', [])
            frame_person_skeletons = []
            if keypoint_data:
                for person_idx, person_keypoints in enumerate(keypoint_data):
                    person_keypoints_np = np.array(person_keypoints, dtype=np.float32)  # [num_nodes, 3]
                    # Pad or trim the number of key-points
                    # if person_keypoints_np.shape[0] < self.num_nodes:
                    #     pad_len = self.num_nodes - person_keypoints_np.shape[0]
                    #     padding = np.zeros((pad_len, 3), dtype=np.float32)
                    #     person_keypoints_np = np.vstack([person_keypoints_np, padding])
                    # elif person_keypoints_np.shape[0] > self.num_nodes:
                    #     person_keypoints_np = person_keypoints_np[:self.num_nodes]
                    person_keypoints_np = person_keypoints_np[:, :2]
                    self.n_features = person_keypoints_np.shape[1]
                    frame_person_skeletons.append(person_keypoints_np)  # Store (x, y)
            all_people_keypoints.append(frame_person_skeletons)

        # Process each person's trajectory independently within the sequence
        max_people = max(len(frame_skeletons) for frame_skeletons in all_people_keypoints) if all_people_keypoints else 0
        processed_sequence = []
        for person_idx in range(max_people):
            person_trajectory = []
            person_centers = []
            person_heights = []
            for frame_idx in range(self.num_frames):
                if person_idx < len(all_people_keypoints[frame_idx]):
                    skeleton = all_people_keypoints[frame_idx][person_idx]  # [N, 2]
                    person_trajectory.append(skeleton)
                    # Calculate geometric center
                    torso_joints = skeleton[self.torso_joint_indices]
                    center_x = np.mean(torso_joints[:, 0])
                    center_y = np.mean(torso_joints[:, 1])
                    person_centers.append(np.array([center_x, center_y], dtype=np.float32))
                    # Calculate bounding box height using all joints
                    min_y = np.min(skeleton[:, 1])
                    max_y = np.max(skeleton[:, 1])
                    height = max_y - min_y
                    person_heights.append(height)
                else:
                    # Pad with zeros if the person is not present in this frame
                    person_trajectory.append(np.zeros((self.num_nodes, 2), dtype=np.float32))
                    person_centers.append(np.zeros(2, dtype=np.float32))
                    person_heights.append(0.0)
            person_trajectory = np.stack(person_trajectory)  # [T, N, 2]
            person_centers = np.stack(person_centers)  # [T, 2]
            person_heights = np.array(person_heights, dtype=np.float32)  # [T]

            normalized_trajectory = []
            for t in range(self.num_frames):
                height = person_heights[t] / 2.0 if person_heights[t] > 0 else 1.0
                center = person_centers[t]
                # Normalize wrt half height
                normalized_frame = (person_trajectory[t] - center) / np.clip(height, a_min=1e-6, a_max=None)
                normalized_trajectory.append(normalized_frame)
            normalized_trajectory = np.stack(normalized_trajectory)  # [T, N, 2]
            processed_sequence.append((person_centers, normalized_trajectory))

        num_people = len(processed_sequence)
        centers = [data[0] for data in processed_sequence]
        normalized_joints = [data[1] for data in processed_sequence]

        padding_needed = self.max_people - num_people
        if padding_needed > 0:
            centers.extend(
                [np.zeros((self.num_frames, self.n_features), dtype=np.float32)] * padding_needed
            )
            normalized_joints.extend(
                [np.zeros((self.num_frames, self.num_nodes, self.n_features), dtype=np.float32)] * padding_needed
            )
        elif padding_needed < 0:
            centers = centers[:self.max_people]
            normalized_joints = normalized_joints[:self.max_people]
        # Prepare input treating each person as a separate entity in the batch dimension.
        padded_high_level_features = np.stack(centers)
        padded_low_level_features = np.stack(normalized_joints)
        # For simplicity, we'll combine geometric centers and normalized joints as features for each node.
        # Each node in the graph will represent a joint of a person at a specific time step.
        # However, the paper describes high-level (person centers) and low-level (normalized joints) graphs.
        # We'll return them separately for now to align with the paper's description. # TODO

        # High-level graph nodes (geometric centers) - [num_people, T, 2]
        high_level_features = torch.tensor(padded_high_level_features,
                                           dtype=torch.float32)  # [1, max_people, T, n_features]
        # Low-level graph nodes (normalized joints) - [num_people, T, N, 2]
        low_level_features =torch.tensor(padded_low_level_features,
                                         dtype=torch.float32)  # [1, max_people, T, N, n_features]
        # The adjacency matrix for the low-level graph (semantic connections between body joints) is fixed.
        # We'll create a simple adjacency based on a typical human skeleton structure.
        # This adjacency will be the same for all people and all time frames in this sample. # TODO
        low_level_adj = self._get_skeleton_adjacency(self.num_nodes)
        # For the high-level graph (connections between people), the adjacency might be based on proximity.
        # This requires information about the spatial arrangement of people, which isn't directly available here.
        # For now, we'll return an empty adjacency or a fully connected one. # TODO
        high_level_adj = self._get_fully_connected_edges(self.max_people) if self.max_people > 0 else torch.empty(
            (2, 0), dtype=torch.long
        )
        return {
            'high_level_features': high_level_features,  # [num_people, T, n_features]
            'low_level_features': low_level_features,  # [num_people, T, N, n_features]
            'low_level_adj': low_level_adj,  # [2, E_low]
            'high_level_adj': high_level_adj  # [2, E_high]
        }

    def _get_fully_connected_edges(self, num_nodes):
        """
        Creates a fully connected adjacency matrix (represented as edge index).
        """
        src, dst = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    src.append(i)
                    dst.append(j)
        return torch.tensor([src, dst], dtype=torch.long)

    def _get_skeleton_adjacency(self, num_nodes):
        """
        Creates a fixed adjacency matrix based on human skeleton connections (HRNet).
        """
        edges = [
            (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9),
            (7, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (7, 16)
        ]
        src = [u for u, v in edges]
        dst = [v for u, v in edges]
        return torch.tensor([src, dst], dtype=torch.long)


if __name__ == '__main__':
    dataset_path = "../data/processed/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
    dataset = SurveillanceAnomalyDataset(root_dir=dataset_path)

    print(f"Number of samples: {len(dataset)}")
    sample = dataset[0]
    print("High-level features shape:", sample['high_level_features'].shape)
    print("Low-level features shape:", sample['low_level_features'].shape)
    print("Low-level adjacency shape:", sample['low_level_adj'].shape)
    print("High-level adjacency shape:", sample['high_level_adj'].shape)

    shapes = np.array([sample['high_level_features'].shape for sample in dataset])
    print(shapes.shape)
    shape, count = np.unique(shapes, axis =0, return_counts=True)
    for i, s in enumerate(shape):
        print(f"{s}: {count[i]}")
