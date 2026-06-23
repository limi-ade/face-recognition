import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter

from loss.losses import *
from nets.iresnet import (iresnet18, iresnet34, iresnet50, iresnet100,
                          iresnet200)
from nets.iresnet2060 import iresnet2060
# from nets.mobilefacenet import get_mbf
# from nets.mobilefacenet_from_mxnet import get_mbf
from nets.mobilefacenet_from_pp import get_mbf

from nets.mobilenet import get_mobilenet
from nets.partial_fc_v2 import PartialFC_V2


class Arcface_Head(Module):
    def __init__(self, num_classes=10575, s=64., m=0.5):
        super(Arcface_Head, self).__init__()
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(num_classes, 512))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(input, F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine.float() > self.th, phi.float(), cosine.float() - self.mm)

        one_hot = torch.zeros(cosine.size()).type_as(phi).long()
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class Arcface(nn.Module):
    def __init__(self, loss, margin_list, interclass_filtering_threshold, embedding_size, num_classes, sample_rate, fp16, backbone, pretrained, distributed, device, **kwargs):
        super(Arcface, self).__init__()
        s = 64
        # loss
        if loss == 'CombinedMarginLoss':
            # CombinedMarginLoss
            margin_loss = eval(loss)(
                s,
                margin_list[0],
                margin_list[1],
                margin_list[2],
                interclass_filtering_threshold
            )
            self.module_partial_fc = PartialFC_V2(margin_loss, embedding_size, num_classes, sample_rate, fp16, distributed, device)
        elif loss == 'ArcFace':
            margin_loss = eval(loss)()
            self.module_partial_fc = PartialFC_V2(margin_loss, embedding_size, num_classes, sample_rate, fp16, distributed, device)

        elif loss == 'CosFace':
            margin_loss = eval(loss)()
            self.module_partial_fc = PartialFC_V2(margin_loss, embedding_size, num_classes, sample_rate, fp16, distributed, device)

        else:
            self.module_partial_fc = Arcface_Head(num_classes=num_classes)
            self.criterion = torch.nn.CrossEntropyLoss()
        
        # backbone
        if backbone == "mbf":
            self.backbone = get_mbf(embedding_size)

        elif backbone == "mobilenetv1":
            self.backbone = get_mobilenet(dropout_keep_prob=0.5, embedding_size=embedding_size)

        elif backbone == "r18":
            self.backbone = iresnet18(dropout_keep_prob=0.5, embedding_size=embedding_size)

        elif backbone == "r34":
            self.backbone = iresnet34(dropout_keep_prob=0.5, embedding_size=embedding_size)

        elif backbone == "r50":
            self.backbone = iresnet50(dropout_keep_prob=0.5, embedding_size=embedding_size)

        elif backbone == "r100":
            self.backbone = iresnet100(dropout_keep_prob=0.5, embedding_size=embedding_size)

        elif backbone == "r200":
            self.backbone = iresnet200(dropout_keep_prob=0.5, embedding_size=embedding_size)

        elif backbone == "r2060":
            self.backbone = iresnet2060(False, num_features=embedding_size, pretrained=False, **kwargs)

        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilefacenet, mobilenetv1.'.format(backbone))

               
    def forward(self, img, local_labels=None, mode="predict"):
        x = self.backbone(img)
        x = x.view(x.size()[0], -1)
        x = F.normalize(x)
        if mode == "predict":
            return x
        else:
            loss_raw = self.module_partial_fc(x, local_labels)
            if isinstance(loss_raw, tuple):
                return loss_raw[0], loss_raw[1]
            else:
                loss = self.criterion(loss_raw, local_labels)   
                return loss_raw, loss
