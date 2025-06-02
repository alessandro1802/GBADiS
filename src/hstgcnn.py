import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import lightning as pl



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
            out_b = []
            for p in range(P):
                node_feat = x[b, p]  # HL: [1, in_channels], LL: [N, in_channels]
                edges = edge_index_list[b][..., :2].permute(2, 1, 0)  # HL: [2, E, 1], LL: [2, E, P]
                out_p = self.gcn(node_feat, edges)  # HL: [1, out_channels], LL: [N, out_channels]
                out_b.append(self.activation(out_p))
            out.append(torch.stack(out_b))  # HL: [P, 1, out_channels], LL: [P, N, out_channels]
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


# Future Frame Predictor
class FFP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=3, num_layers=5, input_frames=4):
        super().__init__()
        layers = []
        current_channels = in_channels
        for i in range(num_layers - 1):
            layers.append(
                nn.Conv2d(current_channels, hidden_channels, kernel_size=(kernel_size, 1), padding=(1, 0)))
            layers.append(nn.PReLU())
            current_channels = hidden_channels
        # Final Conv2d: collapse temporal dim from T-1 to 1
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=(input_frames, 1), padding=(0, 0)))
        self.temporal_layers = nn.Sequential(*layers)
        self.activation = nn.PReLU()
        self.out_channels = out_channels

    def forward(self, x):
        # x: [B, P, T-1, N, in_channels]
        B, P, T_inp, N, C = x.shape
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # [B, P, N, in_channels, T-1]
        x = x.view(B * P * N, C, T_inp, 1)  # [BPN, in_channels, T-1, 1]
        x = self.activation(self.temporal_layers(x))  # [BPN, out_channels, 1, 1]
        x = x.view(B, P, N, self.out_channels).unsqueeze(2)  # [B, P, 1, N, out_channels]
        return x


# Outlier Arbiter
class OA(nn.Module):
    def __init__(self, weights=[1/3, 1/3, 1/3]):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor(weights))  # W₁, W₂, W₃

    def forward(self, x_pred, x_true, f_pred_g, f_true_g):
        """
        Inputs:
        - x_pred: [B, P, 1, N, F_low] - predicted low-level features (5th frame)
        - x_true: [B, P, 1, N, F_low] - ground truth low-level features
        - f_pred_g: [B, P, 1, F_high] - predicted high-level features
        - f_true_g: [B, P, 1, F_high] - ground truth high-level features
        Returns:
        - Anomaly score L: [B]
        """
        B, P, _, N, F_low = x_pred.shape
        _, _, _, F_high = f_pred_g.shape
        # L1: MSE over time, joints, and features
        L1 = ((x_pred - x_true) ** 2).mean(dim=[1, 3, 4])  # [B]
        # L2: max error over time (center point)
        # Assume center point = mean over joints
        center_pred = x_pred.mean(dim=3)  # [B, P, 1, F_low]
        center_true = x_true.mean(dim=3)  # [B, P, 1, F_low]
        center_err = ((center_pred - center_true) ** 2).mean(dim=3).squeeze(2)  # [B, P]
        L2 = center_err.max(dim=1).values  # [B]
        # L3: max pairwise motion vector error over time
        # For all person-pairs i ≠ j, compute: (Δ_pred - Δ_true)^2
        f_pred_g = f_pred_g.squeeze(2)  # [B, P, F_high]
        f_true_g = f_true_g.squeeze(2)  # [B, P, F_high]
        pred_diffs = f_pred_g.unsqueeze(2) - f_pred_g.unsqueeze(1)  # [B, P, P, F_high]
        true_diffs = f_true_g.unsqueeze(2) - f_true_g.unsqueeze(1)  # [B, P, P, F_high]
        motion_err = ((pred_diffs - true_diffs) ** 2).sum(dim=3)  # [B, P, P]
        # Mask diagonal (i == j) by setting it to 0
        eye = torch.eye(P, device=motion_err.device).bool()
        motion_err.masked_fill_(eye.unsqueeze(0), 0.0)
        L3 = motion_err.max(dim=2).values.max(dim=1).values  # [B]
        # Weighted sum
        L = self.weights[0] * L1 + self.weights[1] * L2 + self.weights[2] * L3  # [B]
        return L


# Hierarchical Spatio-Temporal Graph Convolutional Neural Network
class HSTGCNN(pl.LightningModule):
    def __init__(self, high_level_in_channels=2, low_level_in_channels=2,
                 high_level_hidden_channels=32, low_level_hidden_channels=64,
                 fusion_out_channels=128, num_ffp_layers=5, num_oa_branches=3):
        super().__init__()
        self.high_level_stgcnn = STGCNN(high_level_in_channels, high_level_hidden_channels)
        self.low_level_stgcnn = STGCNN(low_level_in_channels, low_level_hidden_channels)
        self.feature_fusion = FeatureFusion(high_level_hidden_channels,
                                            low_level_hidden_channels,
                                            fusion_out_channels)
        self.low_level_ffp = FFP(fusion_out_channels, fusion_out_channels, low_level_in_channels,
                                 num_layers=num_ffp_layers)
        self.oa = OA(low_level_in_channels, num_oa_branches)
        self.loss_fn = nn.MSELoss()
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
        pred = self.low_level_ffp(fused_feat)  # [B, P, 1, N, F_low]
        return pred

    def step(self, batch, batch_idx):
        high_level_features = batch['high_level_features'][:, :, :-1]  # [B, P, T-1, F_high]
        low_level_features = batch['low_level_features'][:, :, :-1]  # [B, P, T-1, N, F_low]
        high_level_adj = batch['high_level_adj'][:, :-1]  # [B, T-1, E_high, 3]
        low_level_adj = batch['low_level_adj'][:, :, :-1]  # [B, P, T-1, E_low, 3]
        future_low_level = batch['low_level_features'][:, :, -1:]  # [B, P, 1, N, F_low]
        pred = self(high_level_features, low_level_features, high_level_adj, low_level_adj)  # [B, P, 1, N, F_low]
        loss = self.loss_fn(pred, future_low_level)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
