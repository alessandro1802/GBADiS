import os
import re
from typing import List

import torch
from torch.utils.data import Dataset
import torch_geometric
from torch_geometric.utils import from_networkx

import numpy as np
import pandas as pd
from rdflib import Graph as RDFGraph
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler


# P: max_people, T: seq_len, N: n_keypoints, F_low: n_low_level_features, F_high: n_high_level_features


class SurveillanceAnomalyDataset(Dataset):
    def __init__(self, root_dir: str,
                 seq_len: int = 5, num_nodes: int = 17, n_features: int = 2, max_people: int = 30, transform=None):
        """
        :param root_dir: path to dataset root containing video folders (e.g., Train001).
        :param seq_len: number of consecutive frames per sequence sample
        :param num_nodes: number of joints per person (N, N=17 based on HRNet).
        :param n_features: number of features per 1 key-point.
        :param max_people: maximum number of people per video.
        :param transform: optional transform to apply to the data.
        """
        self.root_dir = root_dir
        self.T = seq_len
        self.num_nodes = num_nodes
        self.n_features = n_features  # (x, y)
        self.max_people = max_people
        self.transform = transform
        self.samples = self._gather_samples()
        # Indices of the four torso joints for geometric center calculation.
        self.torso_joint_indices = [4, 5, 10, 11]

    def _gather_samples(self):
        """
        Gathers sequences of 'T' consecutive frames from a single CSV per video.
        Each sample contains metadata from T consecutive frames.
        """
        samples = []
        csv_files = [os.path.join(self.root_dir, f) for f in sorted(os.listdir(self.root_dir)) if f.endswith('.csv')]
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            # Group by frame index
            frame_groups = df.groupby("frame_idx")
            frame_indices = sorted(frame_groups.groups.keys())
            # Slide over T-length sequences of frames
            for i in range(len(frame_indices) - self.T + 1):
                frame_seq = frame_indices[i: i + self.T]
                samples.append([frame_groups.get_group(f) for f in frame_seq])  # List of T DataFrames
        return samples

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        frame_data = self.samples[idx]
        trackid_to_skeletons = dict()  # {trackID: [skeleton_t0, skeleton_t1, ..., skeleton_tT]}
        trackid_to_centers = dict()
        trackid_to_heights = dict()
        # Extract the data
        for frame_idx, frame_df in enumerate(frame_data):
            # Iterate detections
            for _, row in frame_df.iterrows():
                track_id = int(row['trackID'])
                # [num_nodes, n_features]
                keypoints = np.array(eval(row['keypoints']), dtype=np.float32)[:, :self.n_features]
                # Initialize new tracks
                if track_id not in trackid_to_skeletons:
                    trackid_to_skeletons[track_id] = [np.zeros((self.num_nodes, self.n_features),
                                                               dtype=np.float32) for _ in range(self.T)]
                    trackid_to_centers[track_id] = [np.zeros(2, dtype=np.float32) for _ in range(self.T)]
                    trackid_to_heights[track_id] = [0.0 for _ in range(self.T)]
                trackid_to_skeletons[track_id][frame_idx] = keypoints
                # Compute geometric center and height
                torso_joints = keypoints[self.torso_joint_indices]
                center = np.mean(torso_joints, axis=0)
                trackid_to_centers[track_id][frame_idx] = center.astype(np.float32)
                height = np.max(keypoints[:, 1]) - np.min(keypoints[:, 1])
                trackid_to_heights[track_id][frame_idx] = height
        # Normalize all tracked sequences
        sequence_centers = []
        sequence_joints = []
        for track_id in trackid_to_skeletons:
            centers = np.stack(trackid_to_centers[track_id])  # [T, 2]
            sequence_centers.append(centers)

            normalized_sequence = []
            trajectory = np.stack(trackid_to_skeletons[track_id])  # [T, num_nodes, n_features]
            heights = np.array(trackid_to_heights[track_id], dtype=np.float32)  # [T]
            for t in range(self.T):
                h = heights[t] / 2.0 if heights[t] > 0 else 1.0
                normalized_frame = (trajectory[t] - centers[t]) / np.clip(h, a_min=1e-6, a_max=None)
                normalized_sequence.append(normalized_frame)
            normalized_sequence = np.stack(normalized_sequence)  # [T, num_nodes, n_features]
            sequence_joints.append(normalized_sequence)
        # Pad/truncate to self.max_people
        num_people = len(sequence_centers)
        padding_needed = self.max_people - num_people
        if padding_needed > 0:
            sequence_centers.extend([np.zeros((self.T, self.n_features), dtype=np.float32)] * padding_needed)
            sequence_joints.extend([np.zeros((self.T, self.num_nodes, self.n_features),
                                             dtype=np.float32)] * padding_needed)
        elif padding_needed < 0:
            sequence_centers = sequence_centers[:self.max_people]
            sequence_joints = sequence_joints[:self.max_people]
        # [max_people, T, 2]
        high_level_features = torch.tensor(np.stack(sequence_centers), dtype=torch.float32)
        # [max_people, T, num_nodes, n_features]
        low_level_features = torch.tensor(np.stack(sequence_joints), dtype=torch.float32)
        # [T, (max_people * (max_people - 1)), 3]
        high_level_adj = self._get_dynamic_center_adjacency(high_level_features)
        # [max_people, T, num_nodes * (num_nodes - 1), 3]
        low_level_adj = self._get_dynamic_joint_adjacency(low_level_features)
        return {
            'high_level_features': high_level_features,  # [max_people, T, 2]
            'low_level_features': low_level_features,  # [max_people, T, num_nodes, n_features]
            'high_level_adj': high_level_adj,  # [T, (max_people * (max_people - 1)), 3]
            'low_level_adj': low_level_adj,  # [max_people, T, num_nodes * (num_nodes - 1), 3]
        }

    def _get_dynamic_center_adjacency(self, centers):
        """
        centers: [max_people, T, 2] - geometric centers
        Returns: [T, E = (max_people * (max_people - 1)), 3] - edge list for each frame: (i, j, weight)
        """
        max_people, T, _ = centers.shape
        edges_per_time = []
        for t in range(T):
            edge_list = []
            frame_centers = centers[:, t]  # [max_people, 2]
            for i in range(max_people):
                for j in range(max_people):
                    if i == j:
                        continue
                    dist_sq = torch.sum((frame_centers[i] - frame_centers[j]) ** 2).item()
                    weight = 1.0 / dist_sq if dist_sq > 1e-6 else 0.0
                    edge_list.append([i, j, weight])
            edges_per_time.append(edge_list)
        return torch.tensor(edges_per_time, dtype=torch.float32)  # [T, E, 3]

    def _get_dynamic_joint_adjacency(self, joints):
        """
        joints: [max_people, T, num_nodes, n_features] - normalized joint positions
        Returns: [max_people, T, E = (num_nodes * (num_nodes - 1)), 3] - each edge (i, j, weight)
        """
        max_people, T, num_nodes, n_features = joints.shape
        edges_per_person = []

        for person_idx in range(max_people):
            person_edges = []
            for t in range(T):
                frame_joints = joints[person_idx, t]  # [num_nodes, n_features]
                edge_list = []
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i == j:
                            continue
                        dist_sq = torch.sum((frame_joints[i] - frame_joints[j]) ** 2).item()
                        weight = 1.0 / dist_sq if dist_sq > 1e-6 else 0.0
                        edge_list.append([i, j, weight])
                person_edges.append(edge_list)  # shape: [E, 3]
            edges_per_person.append(person_edges)
        return torch.tensor(edges_per_person, dtype=torch.float32)  # [max_people, T, E, 3]


def load_and_process_ontology(ontology_path: str, sentence_transformer: str = "all-MiniLM-L6-v2",
                              device="mps") -> torch_geometric.data.Data:
    def _extract_label(node: str) -> str:
        node_str = str(node)
        # Handle URI-based nodes (e.g., http://example.org/ChannelizedIntersection)
        if "#" in node_str:
            return node_str.split("#")[-1]
        elif "/" in node_str:
            return node_str.split("/")[-1]
        # Handle blank nodes or UUID-style strings
        if node_str.startswith("N") and len(node_str) > 20:
            return ""  # skip or mark as unknown
        return node_str

    # Load ontology
    rdf_graph = RDFGraph()
    rdf_graph.parse(ontology_path, format="xml")
    # Build a directed graph from RDF
    G = nx.DiGraph()
    for subj, pred, obj in rdf_graph:
        G.add_edge(str(subj), str(obj), relation=str(pred))
    # Preprocess node labels
    labels = []
    for node in G.nodes():
        label = _extract_label(node)
        if label.strip():
            labels.append(label)
        else:
            labels.append("unknown")  # Placeholder for blank nodes
    # Embed nodes
    model = SentenceTransformer(sentence_transformer, device=device)
    embeddings = model.encode(labels, convert_to_tensor=True)
    for i, node in enumerate(G.nodes()):
        G.nodes[node]["x"] = embeddings[i]
    return from_networkx(G).to(device)


def load_UCSD_labels(filepath: str, video_lens: List[int]) -> np.ndarray:
    # Load .m file
    gt_list = []
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(r'gt_frame\s*=\s*\[(.*?)\];', line)
            if match:
                frame_str = match.group(1)
                # Expand MATLAB-style ranges (60:152 -> 60, 61, ..., 152)
                frame_nums = []
                for part in frame_str.replace(',', ' ').split():
                    if ':' in part:
                        start, end = map(int, part.split(':'))
                        frame_nums.extend(range(start, end + 1))
                    else:
                        frame_nums.append(int(part))
                # Convert to 0-based indexing
                gt_list.append([i - 1 for i in frame_nums])
    # Create binary labels for each frame in every video
    labels = []
    for i, gt_frames in enumerate(gt_list):
        label = np.zeros(video_lens[i], dtype=int)
        label[gt_frames] = 1
        labels.append(label)
    return labels


def normalize_scores(scores: List[float], scaler=MinMaxScaler) -> np.ndarray:
    scores_arr = np.array(scores)
    scores_arr = scaler().fit_transform(scores_arr[..., np.newaxis])
    return scores_arr / np.amax(scores_arr)


def reduce_to_regions(labels: List[float], predictions: List[float]) -> (np.ndarray, np.ndarray):
    labels = np.array(labels)
    predictions = np.array(predictions)
    reduced_labels = []
    reduced_preds = []
    start = 0
    while start < len(labels):
        # Find start of next region (either label or prediction changes)
        end = start + 1
        while end < len(labels) and labels[end] == labels[start]:
            end += 1
        # Reduce current region
        region_label = labels[start]
        region_max_pred = predictions[start:end].max()
        reduced_labels.append(region_label)
        reduced_preds.append(region_max_pred)
        start = end
    return np.array(reduced_labels), np.array(reduced_preds)


if __name__ == '__main__':
    dataset_path = "../data/processed/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
    dataset = SurveillanceAnomalyDataset(root_dir=dataset_path)

    print(f"Number of samples: {len(dataset)}")
    sample = dataset[0]
    print("High-level features shape:", sample['high_level_features'].shape)
    print("Low-level features shape:", sample['low_level_features'].shape)
    print("High-level adjacency shape:", sample['high_level_adj'].shape)
    print("Low-level adjacency shape:", sample['low_level_adj'].shape)
