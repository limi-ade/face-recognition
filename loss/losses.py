import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class CombinedMarginLoss(torch.nn.Module):
    def __init__(self, 
                 s, 
                 m1,
                 m2,
                 m3,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold
        
        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False


    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty    
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            with torch.no_grad():
                target_logit.arccos_()
                logits.arccos_()
                final_target_logit = target_logit + self.m2
                logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
                logits.cos_()
            logits = logits * self.s        

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise

        return logits

class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s   
        return logits


class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits



class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3, squared=False):
        """
        margin: 正样本距离与负样本距离之间的最小间隔
        squared: 是否使用欧氏距离的平方 (推荐 False，与验证时一致)
        """
        super().__init__()
        self.margin = margin
        self.squared = squared

    def forward(self, embeddings, labels):
        """
        embeddings: (B, D) 已经过 L2 归一化的特征向量，||x||=1
        labels: (B,)
        返回: 标量损失
        """
        # 计算成对欧氏距离矩阵 (B, B)
        # 由于 embedding 已归一化，欧氏距离范围 [0,2]
        dist_mat = torch.cdist(embeddings, embeddings, p=2)  # (B, B)
        if self.squared:
            dist_mat = dist_mat.pow(2)

        # 构造同身份/不同身份的掩码
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)   # (B, B)
        labels_ne = ~labels_eq

        # 每个 anchor 的最难正样本（距离最大）
        dist_pos = dist_mat.clone()
        dist_pos[labels_ne] = -1.0  # 屏蔽不同身份
        hardest_pos_dist, _ = dist_pos.max(dim=1, keepdim=True)  # (B, 1)

        # 每个 anchor 的最难负样本（距离最小）
        dist_neg = dist_mat.clone()
        dist_neg[labels_eq] = float('inf')  # 屏蔽同身份
        hardest_neg_dist, _ = dist_neg.min(dim=1, keepdim=True)  # (B, 1)

        # 三元组损失 (Relu 防止负距离差)
        loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
        return loss.mean()