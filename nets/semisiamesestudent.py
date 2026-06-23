import copy

import torch
import torch.nn as nn



class SemiSiameseStudent(nn.Module):
    def __init__(self, backbone, feature_dim=512, pred_hidden=256):
        super().__init__()
        # Online 分支（可训练）
        self.online_encoder = backbone
        self.online_projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.online_predictor = nn.Sequential(
            nn.Linear(feature_dim, pred_hidden),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(),
            nn.Linear(pred_hidden, feature_dim)
        )
        # Target 分支（EMA更新）
        
        self.target_encoder = copy.deepcopy(backbone)
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        self.target_projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.target_projector.load_state_dict(self.online_projector.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    def forward(self, x):
        # 返回 Online 特征和预测（用于自监督）
        f = self.online_encoder(x)
        z = self.online_projector(f)
        p = self.online_predictor(z)
        return p

    def forward_target(self, x):
        # 返回 Target 特征
        with torch.no_grad():
            f = self.target_encoder(x)
            z = self.target_projector(f)
        return z

    def update_target(self, tau=0.996):
        # EMA 更新 Target
        with torch.no_grad():
            for p_online, p_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
                p_target.data = tau * p_target.data + (1 - tau) * p_online.data
            for p_online, p_target in zip(self.online_projector.parameters(), self.target_projector.parameters()):
                p_target.data = tau * p_target.data + (1 - tau) * p_online.data