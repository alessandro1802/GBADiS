import os
import glob
import json

import torch
from torch.utils.data import Dataset
import torch_geometric
from torch_geometric.utils import from_networkx

import numpy as np
from rdflib import Graph as RDFGraph
import networkx as nx
from sentence_transformers import SentenceTransformer


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
        Gathers paths to JSON files for each sequence in the dataset.
        Each sample will contain a sequence of 'T' JSON files.
        """
        samples = []
        for seq_folder in sorted(os.listdir(self.root_dir)):
            seq_path = os.path.join(self.root_dir, seq_folder)
            json_dir = os.path.join(seq_path, 'json')
            if os.path.isdir(json_dir):
                json_files = sorted(glob.glob(os.path.join(json_dir, '*.json')))
                if len(json_files) >= self.T:
                    for i in range(len(json_files) - self.T + 1):
                        samples.append(json_files[i : i + self.T])
        return samples

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths = self.samples[idx]
        trackid_to_skeletons = dict()  # {trackID: [skeleton_t0, skeleton_t1, ..., skeleton_tT]}
        trackid_to_centers = dict()
        trackid_to_heights = dict()
        # Read the data
        for frame_idx, frame_path in enumerate(frame_paths):
            with open(frame_path, 'r') as f:
                data = json.load(f)
            tracks = data.get('tracks', [])  # [[trackID, box], ...]
            keypoint_data = data.get('keypoints', [])  # [[num_nodes, 3], ...]
            # Map trackID to keypoints
            for (track_id, _), keypoints in zip(tracks, keypoint_data):
                keypoints_arr = np.array(keypoints, dtype=np.float32)[:, :self.n_features]  # [num_nodes, n_features]
                if track_id not in trackid_to_skeletons:
                    trackid_to_skeletons[track_id] = []
                    trackid_to_centers[track_id] = []
                    trackid_to_heights[track_id] = []
                trackid_to_skeletons[track_id].append(keypoints_arr)
                # Compute geometric center and height
                torso_joints = keypoints_arr[self.torso_joint_indices]
                center = np.mean(torso_joints, axis=0)
                height = np.max(keypoints_arr[:, 1]) - np.min(keypoints_arr[:, 1])
                trackid_to_centers[track_id].append(center.astype(np.float32))
                trackid_to_heights[track_id].append(height)
            # Pad missing trackIDs with zeros for this frame
            for track_id in trackid_to_skeletons:
                while len(trackid_to_skeletons[track_id]) < frame_idx + 1:
                    trackid_to_skeletons[track_id].append(np.zeros((self.num_nodes, self.n_features), dtype=np.float32))
                    trackid_to_centers[track_id].append(np.zeros(2, dtype=np.float32))
                    trackid_to_heights[track_id].append(0.0)
        # Normalize all tracked sequences
        sequence_centers = []
        sequence_joints = []
        for track_id in trackid_to_skeletons:
            centers = np.stack(trackid_to_centers[track_id])  # [T, 2]
            sequence_centers.append(centers)
            trajectory = np.stack(trackid_to_skeletons[track_id])  # [T, num_nodes, n_features]
            heights = np.array(trackid_to_heights[track_id], dtype=np.float32)  # [T]

            normalized = []
            for t in range(self.T):
                h = heights[t] / 2.0 if heights[t] > 0 else 1.0
                normalized_frame = (trajectory[t] - centers[t]) / np.clip(h, a_min=1e-6, a_max=None)
                normalized.append(normalized_frame)
            normalized = np.stack(normalized)  # [T, num_nodes, n_features]
            sequence_joints.append(normalized)
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


if __name__ == '__main__':
    dataset_path = "../data/processed/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
    dataset = SurveillanceAnomalyDataset(root_dir=dataset_path)

    print(f"Number of samples: {len(dataset)}")
    sample = dataset[0]
    print("High-level features shape:", sample['high_level_features'].shape)
    print("Low-level features shape:", sample['low_level_features'].shape)
    print("High-level adjacency shape:", sample['high_level_adj'].shape)
    print("Low-level adjacency shape:", sample['low_level_adj'].shape)
