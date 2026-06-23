import torch
import torch.nn as nn

from .mobilefacenet_from_pp import MobileFaceNet


class MobileFaceNetWithAdapter(nn.Module):
    def __init__(self, feature_dim=512, base_feature_dim=128):
        super().__init__()
        self.base_model = MobileFaceNet(feature_dim=base_feature_dim)
        self.adapter = nn.Linear(base_feature_dim, feature_dim)

    def forward(self, x):
        x = self.base_model(x)
        x = self.adapter(x)
        return x


def get_mbf_with_adapter(embedding_size=512, pretrained=None):
    if pretrained:
        raise ValueError("No pretrained model for mobilefacenet_with_adapter")
    return MobileFaceNetWithAdapter(feature_dim=embedding_size)
