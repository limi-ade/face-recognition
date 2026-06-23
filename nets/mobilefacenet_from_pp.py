import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super().__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        return self.prelu(x)


class BottleNeck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super().__init__()
        self.connect = stride == 1 and inp == oup

        hidden_dim = inp * expansion
        self.conv = nn.Sequential(
            # 1x1 expand
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(hidden_dim),

            # 3x3 depthwise
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(hidden_dim),

            # 1x1 project
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        return self.conv(x)


MobileFaceNet_BottleNeck_Setting = [
    # t, c, n, s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]


class MobileFaceNet(nn.Module):
    def __init__(self, feature_dim=128, bottleneck_setting=MobileFaceNet_BottleNeck_Setting):
        super().__init__()

        self.conv1 = ConvBlock(3, 64, 3, 2, 1)
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.cur_channel = 64
        self.blocks = self._make_layer(BottleNeck, bottleneck_setting)

        self.conv2 = ConvBlock(128, 512, 1, 1, 0)
        self.linear7 = ConvBlock(512, 512, 7, 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, feature_dim, 1, 1, 0, linear=True)

        self.adapter = nn.Linear(feature_dim, 512)

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.cur_channel, c, s, t))
                else:
                    layers.append(block(self.cur_channel, c, 1, t))
                self.cur_channel = c
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)     # (N, 512, 1, 1)
        x = self.linear1(x)     # (N, feat_dim, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.adapter(x)
        return x


def get_mbf(embedding_size, pretrained=None):
    if pretrained:
        raise ValueError("No pretrained model for mobilefacenet")
    return MobileFaceNet(embedding_size)