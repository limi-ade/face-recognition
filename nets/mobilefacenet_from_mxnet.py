import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    """倒置残差块：1×1扩张 → 3×3深度卷积 → 1×1投影，可加残差"""
    def __init__(self, in_ch, out_ch, expansion, stride):
        super().__init__()
        self.stride = stride
        self.use_shortcut = (stride == 1) and (in_ch == out_ch)
        hidden_ch = in_ch * expansion

        self.conv1 = nn.Conv2d(in_ch, hidden_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_ch, momentum=0.9, eps=1e-3)
        self.prelu1 = nn.PReLU(hidden_ch)

        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, 3, stride, 1,
                               groups=hidden_ch, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_ch, momentum=0.9, eps=1e-3)
        self.prelu2 = nn.PReLU(hidden_ch)

        self.conv3 = nn.Conv2d(hidden_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch, momentum=0.9, eps=1e-3)
        # 注意：原版投影后无激活

    def forward(self, x):
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.prelu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.use_shortcut:
            out = out + x
        return out


class MobileFaceNetY1(nn.Module):
    def __init__(self, emb_size=128):
        super().__init__()

        # Stage 1
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.9, eps=1e-3)
        self.prelu1 = nn.PReLU(64)

        # Stage 2 (conv_2_dw)
        self.conv2_dw = nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False)
        self.bn2_dw = nn.BatchNorm2d(64, momentum=0.9, eps=1e-3)
        self.prelu2_dw = nn.PReLU(64)

        # Stage 3 (dconv_23 + 4*res_3)
        self.stage3 = nn.ModuleList()
        self.stage3.append(Bottleneck(64, 64, expansion=2, stride=2))   # dconv_23
        for _ in range(4):
            self.stage3.append(Bottleneck(64, 64, expansion=2, stride=1))

        # Stage 4 (dconv_34 + 6*res_4)
        self.stage4 = nn.ModuleList()
        self.stage4.append(Bottleneck(64, 128, expansion=4, stride=2))  # dconv_34
        for _ in range(6):
            self.stage4.append(Bottleneck(128, 128, expansion=2, stride=1))

        # Stage 5 (dconv_45 + 2*res_5)
        self.stage5 = nn.ModuleList()
        self.stage5.append(Bottleneck(128, 128, expansion=4, stride=2)) # dconv_45
        for _ in range(2):
            self.stage5.append(Bottleneck(128, 128, expansion=2, stride=1))

        # conv_6sep
        self.conv6sep = nn.Conv2d(128, 512, 1, bias=False)
        self.bn6sep = nn.BatchNorm2d(512, momentum=0.9, eps=1e-3)
        self.prelu6sep = nn.PReLU(512)

        # GDC
        self.gdc_conv = nn.Conv2d(512, 512, 7, groups=512, bias=False)
        self.gdc_bn = nn.BatchNorm2d(512, momentum=0.9, eps=1e-3)

        # 嵌入
        self.fc = nn.Linear(512, emb_size, bias=True)       # 有偏置！
        self.fc_bn = nn.BatchNorm1d(emb_size, momentum=0.9, affine=True, eps=2e-5)            # fix_gamma，但保留参数值

    def forward(self, x):
        # 输入应为 [0,255] 浮点 RGB
        x = (x - 127.5) * 0.0078125

        x = self.prelu1(self.bn1(self.conv1(x)))            # 56
        x = self.prelu2_dw(self.bn2_dw(self.conv2_dw(x)))   # 56

        for layer in self.stage3:
            x = layer(x)                                    # 28→28
        for layer in self.stage4:
            x = layer(x)                                    # 14→14
        for layer in self.stage5:
            x = layer(x)                                    # 7→7

        x = self.prelu6sep(self.bn6sep(self.conv6sep(x)))   # 7×7×512
        x = self.gdc_bn(self.gdc_conv(x))                   # 1×1×512
        x = x.view(x.size(0), -1)                           # (N, 512)
        x = self.fc(x)
        x = self.fc_bn(x)
        return x
    


def get_mbf(embedding_size, pretrained=None):
    if pretrained:
        raise ValueError("No pretrained model for mobilefacenet")
    return MobileFaceNetY1(embedding_size)