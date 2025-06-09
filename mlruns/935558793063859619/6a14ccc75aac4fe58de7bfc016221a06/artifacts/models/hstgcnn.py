from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import from_networkx
import lightning as pl

from rdflib import Graph as RDFGraph
from rdflib import RDF, RDFS, OWL
import networkx as nx


# B: batch_size = 16, P: max_pople = 30, T: seq_len = 5, N: n_keypoints = 17
# F_low: n_low_level_features = 2, F_high: n_high_level_features = 2
# E_high: n_high_level_edges = P * (P-1) = 870, E_low: n_low_level_edges = n_keypoints * (n_keypoints-1) = 272


# Spatio-Temporal Graph Convolutional Neural Network
class STGCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.activation = nn.PReLU()

    def forward(self, x, edge_index_list):
        # HL - x: [B, P, T[i], in_channels], edge_index_list: [B, 1, E, 3]
        # LL - x: [B, P, N,    in_channels], edge_index_list: [B, P, E, 3]
        B, P, _, _ = x.shape
        out = []
        for b in range(B):
            # HL
            if edge_index_list.shape[1] == 1:
                node_feat = x[b, :, 0, :]   # [P, in_channels]
                edge_index = edge_index_list[b, 0, :, :2].T.contiguous()  # [2, E]
                out_b = self.activation(self.gcn(node_feat, edge_index))  # [P, out_channels]
                out_b = out_b.unsqueeze(1)
                out.append(out_b)  # [P, 1, out_channels]
            # LL
            else:
                out_b = []
                for p in range(P):
                    node_feat = x[b, p]  # [N, in_channels]
                    edge_index = edge_index_list[b, p, :, :2].T.contiguous()  # [2, E]
                    out_b.append(self.activation(self.gcn(node_feat, edge_index))) # [N, out_channels]
                out.append(torch.stack(out_b))  # [P, N, out_channels]
        return torch.stack(out)  # HL: [B, P, 1, out_channels], LL: [B, P, N, out_channels]


class FeatureFusion(nn.Module):
    def __init__(self, high_level_out_channels, low_level_out_channels, fusion_out_channels):
        super().__init__()
        self.fc = nn.Linear(high_level_out_channels + low_level_out_channels, fusion_out_channels)
        self.activation = nn.PReLU()

    def forward(self, high_level_feat, low_level_feat):
        # HL: [B, P, T-1, high_level_out_channels], LL: [B, P, T-1, N, low_level_out_channels]
        _, _, _, N, _ = low_level_feat.shape
        # [B, P, T-1, N, high_level_out_channels]
        high_level_expanded = high_level_feat.unsqueeze(3).expand(-1, -1, -1, N, -1)
        # [B, P, T-1, N, low_level_out_channels + high_level_out_channels]
        fused_feat = torch.cat([low_level_feat, high_level_expanded], dim=-1)
        return self.activation(self.fc(fused_feat)) # [B, P, T-1, N, fusion_out_channels]


class KnowledgeFeatureFusion(nn.Module):
    def __init__(self, ontology_path, high_level_out_channels, low_level_out_channels,
                 gnn_hidden=32, fusion_out_channels=128, device="mps"):
        super().__init__()
        # Load and process ontology
        pyg_data = self.load_and_process_ontology(ontology_path, device)
        self.x = pyg_data.x
        self.edge_index = pyg_data.edge_index
        # GNN to extract ontology representation
        self.gnn1 = GCNConv(pyg_data.num_node_features, gnn_hidden)
        self.gnn2 = GCNConv(gnn_hidden, gnn_hidden)
        self.pool = global_mean_pool  # Aggregates to [1, gnn_hidden]
        # Final fusion: [hl+ll+ontology] → fusion_out
        self.fusion = nn.Sequential(nn.Linear(high_level_out_channels + low_level_out_channels + gnn_hidden,
                                              fusion_out_channels),
                                    nn.PReLU())

    def load_and_process_ontology(self, ontology_path, device):
        rdf_graph = RDFGraph()
        rdf_graph.parse(ontology_path, format="xml")
        # Build a directed graph from RDF
        G = nx.DiGraph()
        for subj, pred, obj in rdf_graph:
            G.add_edge(str(subj), str(obj), relation=str(pred))
        # Add node features
        label_count = defaultdict(int)
        for node in G.nodes():
            subj = node
            # Semantic roles
            is_class = (subj, RDF.type, OWL.Class) in rdf_graph or (subj, RDF.type, RDFS.Class) in rdf_graph
            is_property = (subj, RDF.type, RDF.Property) in rdf_graph
            # Degrees
            in_deg = G.in_degree(node)
            out_deg = G.out_degree(node)
            total_deg = in_deg + out_deg
            # Label presence
            has_label = (subj, RDFS.label, None) in rdf_graph
            label_count[len(list(rdf_graph.objects(subj, RDFS.label)))] += 1
            # Number of distinct relations
            predicates = set(pred for s, pred, o in rdf_graph.triples((subj, None, None)))
            num_relations = len(predicates)
            # Whether it’s defined as a subclass
            is_subclass = (subj, RDFS.subClassOf, None) in rdf_graph
            # Feature vector
            feature_vector = [
                float(is_class),
                float(is_property),
                float(is_subclass),
                float(has_label),
                float(in_deg),
                float(out_deg),
                float(total_deg),
                float(num_relations),
            ]
            G.nodes[node]["x"] = torch.tensor(feature_vector, dtype=torch.float)
        return from_networkx(G).to(device)

    def forward(self, high_level_feat, low_level_feat):
        # HL: [B, P, T-1, high_level_out_channels], LL: [B, P, T-1, N, low_level_out_channels]
        B, P, T_inp, N, _ = low_level_feat.shape
        # Ontology embedding
        h = F.relu(self.gnn1(self.x, self.edge_index))
        h = F.relu(self.gnn2(h, self.edge_index))
        batch_tensor_for_pool = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        h_pooled = self.pool(h, batch_tensor_for_pool)  # [1, gnn_hidden]
        h_onto = h_pooled.expand(B, P, T_inp, N, -1)  # [B, P, T-1, N, gnn_hidden]
        # Expand HL features to match [B, P, T-1, N, high_level_out_channels]
        high_level_exp = high_level_feat.unsqueeze(3).expand(-1, -1, -1, N, -1)
        # Fusion -> [B, P, T-1, N, high_level_out_channels+low_level_out_channels+gnn_hidden]
        fused = torch.cat([low_level_feat, high_level_exp, h_onto], dim=-1)
        return self.fusion(fused)  # [B, P, T-1, N, fusion_out_channels]


# Future Frame Predictor
class FFP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_high, out_low, kernel_size=3, num_layers=5, input_frames=4):
        super().__init__()
        # Shared temporal backbone
        layers = []
        current_channels = in_channels
        for i in range(num_layers - 1):
            layers.append(
                nn.Conv2d(current_channels, hidden_channels, kernel_size=(kernel_size, 1), padding=(1, 0)))
            layers.append(nn.PReLU())
            current_channels = hidden_channels
        self.shared_backbone = nn.Sequential(*layers)
        # Low-level head
        self.low_out = nn.Conv2d(hidden_channels, out_low, kernel_size=(input_frames, 1), padding=(0, 0))
        # High-level head
        self.high_out = nn.Conv2d(hidden_channels, out_high, kernel_size=(input_frames, 1), padding=(0, 0))
        self.activation = nn.PReLU()

    def forward(self, x):
        # x: [B, P, T-1, N, in_channels]
        B, P, T_inp, N, C = x.shape
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # [B, P, N, in_channels, T-1]
        x = x.view(B * P * N, C, T_inp, 1)  # [BPN, in_channels, T-1, 1]
        x = self.activation(self.shared_backbone(x))  # [BPN, hidden_channels, T-1, 1]
        # Low-level prediction (per keypoint)
        low_out = self.low_out(x)  # [BPN, F_low, 1, 1]
        low_out = low_out.view(B, P, N, -1).unsqueeze(2)  # [B, P, 1, N, out_low]
        # High-level prediction (pooled across N)
        x_pooled = x.view(B, P, N, -1, T_inp, 1).mean(dim=2).view(B * P, -1, T_inp, 1)  # [BP, hidden_channels, T-1, 1]
        high_out = self.high_out(x_pooled)  # [BP, out_high, 1, 1]
        high_out = high_out.view(B, P, 1, -1)  # [B, P, 1, out_high]
        return low_out, high_out


# Outlier Arbiter
class OA(nn.Module):
    def __init__(self, weights=[1/3, 1/3, 1/3]):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))  # W1, W2, W3

    def forward(self, f_pred_g, f_true_g, f_pred_l, f_true_l):
        """
        Inputs:
        - f_pred_g: [B, P, 1, F_high] high-level predictions (e.g., center or motion)
        - f_true_g: [B, P, 1, F_high] high-level ground truth
        - f_pred_l: [B, P, 1, N, F_low] low-level prediction (per individual pose)
        - f_true_l: [B, P, 1, N, F_low] low-level ground truth
        """
        # L1: MSE over people, time, joints, and features
        L1 = ((f_pred_l - f_true_l) ** 2).mean(dim=[1, 2, 3, 4])  # [B]
        # L2: max error over time for each person, then mean over people
        per_person_max_error = ((f_pred_g - f_true_g) ** 2).view(f_pred_g.shape[0],
                                                                 f_pred_g.shape[1],
                                                                 -1).max(dim=2).values  # [B, P]
        L2 = per_person_max_error.mean(dim=1)  # [B]
        # # L3: max pairwise motion vector error over time for each person
        # motion_pred = f_pred_g[:, :, 1:, :] - f_pred_g[:, :, :-1, :]  # [B, P, T-1, F_high]
        # motion_true = f_true_g[:, :, 1:, :] - f_true_g[:, :, :-1, :]  # [B, P, T-1, F_high]
        # motion_error = ((motion_pred - motion_true) ** 2).sum(dim=3)  # [B, P, T-1]
        # max_motion_error = motion_error.max(dim=2).values  # [B, P]
        # L3 = max_motion_error.mean(dim=1)  # [B]
        # Weighted sum of losses
        L = self.weights[0] * L1 + self.weights[1] * L2 # + self.weights[2] * L3  # [B]
        return L


# Hierarchical Spatio-Temporal Graph Convolutional Neural Network
class HSTGCNN(pl.LightningModule):
    def __init__(self, high_level_in_channels=2, low_level_in_channels=2,
                 high_level_hidden_channels=32, low_level_hidden_channels=64,
                 fusion_out_channels=128, num_ffp_layers=5, oa_weights=[1/3, 1/3, 1/3], lr=1e-4):
        super().__init__()
        self.high_level_stgcnn = STGCNN(high_level_in_channels, high_level_hidden_channels)
        self.low_level_stgcnn = STGCNN(low_level_in_channels, low_level_hidden_channels)
        self.feature_fusion = FeatureFusion(high_level_hidden_channels,
                                            low_level_hidden_channels,
                                            fusion_out_channels)
        self.ffp = FFP(fusion_out_channels, fusion_out_channels,
                       high_level_in_channels, low_level_in_channels,
                       num_layers=num_ffp_layers)
        self.oa = OA(oa_weights)
        self.loss_fn = nn.MSELoss()
        self.learning_rate = lr
        self.save_hyperparameters()

    def forward(self, high_level_features, low_level_features, high_level_adj, low_level_adj):
        # high_level_features: [B, P, T-1, F_high], high_level_adj: [B, T-1, E_high, 3]
        # low_level_features: [B, P, T-1, N, F_low], low_level_adj: [B, P, T-1, E_low, 3]
        _, _, _, F_high = high_level_features.shape
        B, P, T_inp, N, F_low = low_level_features.shape
        high_level_features = high_level_features.permute(0, 2, 1, 3).contiguous()  # [B, T-1, P, F_high]
        low_level_features = low_level_features.permute(0, 2, 1, 3, 4).contiguous()  # [B, T-1, P, N, F_low]
        # Prepare inputs for STGCNN
        hl_feats_per_t = []
        ll_feats_per_t = []
        for t in range(T_inp):
            hl_input_t = high_level_features[:, t, :, :].unsqueeze(2)  # [B, P, 1, F_high]
            hl_edges_t = high_level_adj[:, t, :, :].unsqueeze(1).int()  # [B, 1, E_high, 3]
            hl_feat_t = self.high_level_stgcnn(hl_input_t, hl_edges_t).squeeze(2)  # [B, P, C_high]
            hl_feats_per_t.append(hl_feat_t)

            ll_input_t = low_level_features[:, t, :, :, :]  # [B, P, N, F_low]
            ll_edges_t = low_level_adj[:, :, t, :, :].int()  # [B, P, E_low, 3]
            ll_feat_t = self.low_level_stgcnn(ll_input_t, ll_edges_t)  # [B, P, N, C_low]
            ll_feats_per_t.append(ll_feat_t)
        # Stack over time
        hl_feat = torch.stack(hl_feats_per_t, dim=2)  # [B, P, T-1, C_high]
        ll_feat = torch.stack(ll_feats_per_t, dim=2)  # [B, P, T-1, N, C_low]
        # Feature Fusion
        fused_feat = self.feature_fusion(hl_feat, ll_feat)  # [B, P, T-1, N, fusion_channels]
        # Predict
        ll_pred, hl_pred = self.ffp(fused_feat)  # [B, P, 1, N, F_low], [B, P, 1, F_high]
        return hl_pred, ll_pred,

    def step(self, batch):
        high_level_features = batch['high_level_features'][:, :, :-1]  # [B, P, T-1, F_high]
        low_level_features = batch['low_level_features'][:, :, :-1]  # [B, P, T-1, N, F_low]
        high_level_adj = batch['high_level_adj'][:, :-1]  # [B, T-1, E_high, 3]
        low_level_adj = batch['low_level_adj'][:, :, :-1]  # [B, P, T-1, E_low, 3]

        future_high_level = batch['high_level_features'][:, :, -1:]  # [B, P, 1, F_high]
        future_low_level = batch['low_level_features'][:, :, -1:]  # [B, P, 1, N, F_low]

        hl_pred, ll_pred = self(high_level_features, low_level_features,
                                high_level_adj, low_level_adj)  # [B, P, 1, N, F_low]
        return hl_pred, future_high_level, ll_pred, future_low_level

    def training_step(self, batch):
        f_pred_g, f_true_g, f_pred_l, f_true_l = self.step(batch)
        loss_hl = self.loss_fn(f_pred_g, f_true_g)
        loss_ll = self.loss_fn(f_pred_l, f_true_l)
        loss = loss_ll + loss_hl
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        f_pred_g, f_true_g, f_pred_l, f_true_l = self.step(batch)
        loss_hl = self.loss_fn(f_pred_g, f_true_g)
        loss_ll = self.loss_fn(f_pred_l, f_true_l)
        loss = loss_ll + loss_hl
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def predict(self, batch):
        f_pred_g, f_true_g, f_pred_l, f_true_l = self.step(batch)
        anomaly_score = self.oa(f_pred_g, f_true_g, f_pred_l, f_true_l)
        return anomaly_score

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class KnowledgeHSTGCNN(HSTGCNN):
    def __init__(self, ontology_path, ontology_device="mps",
                 high_level_in_channels=2, low_level_in_channels=2,
                 high_level_hidden_channels=32, low_level_hidden_channels=64,
                 knowledge_gnn_hidden=32, fusion_out_channels=128,
                 num_ffp_layers=5, oa_weights=[1/3, 1/3, 1/3], lr=1e-4):
        super().__init__()
        self.high_level_stgcnn = STGCNN(high_level_in_channels, high_level_hidden_channels)
        self.low_level_stgcnn = STGCNN(low_level_in_channels, low_level_hidden_channels)
        self.feature_fusion = KnowledgeFeatureFusion(ontology_path,
                                                     high_level_hidden_channels, low_level_hidden_channels,
                                                     knowledge_gnn_hidden, fusion_out_channels, ontology_device)
        self.ffp = FFP(fusion_out_channels, fusion_out_channels,
                       high_level_in_channels, low_level_in_channels,
                       num_layers=num_ffp_layers)
        self.oa = OA(oa_weights)
        self.loss_fn = nn.MSELoss()
        self.learning_rate = lr
        self.save_hyperparameters()
