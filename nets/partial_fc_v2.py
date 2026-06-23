import math
from typing import Callable

import torch
from torch import distributed as dist
from torch.nn.functional import linear, normalize


class PartialFC_V2(torch.nn.Module):
    _version = 2

    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        sample_rate: float = 1.0,
        fp16: bool = False,
        distributed: bool = False,
        device: str = 'cuda'
    ):
        super(PartialFC_V2, self).__init__()
        self.distributed = distributed
        self.device = device
        self.fp16 = fp16 and self.device == 'cuda'   # CPU 不支持 fp16 训练

        # 仅在多卡分布式且非 CPU 时初始化进程组（这里假设外部已经初始化）
        # 如果外部未初始化，单卡时 world_size=1 无需 init
        if distributed and device == 'cuda':
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.dist_cross_entropy = DistCrossEntropy(self.world_size)

        self.embedding_size = embedding_size
        self.sample_rate = sample_rate
        self.num_local = num_classes // self.world_size + int(
            self.rank < num_classes % self.world_size
        )
        self.class_start = num_classes // self.world_size * self.rank + min(
            self.rank, num_classes % self.world_size
        )
        self.num_sample = int(self.sample_rate * self.num_local)
        self.last_batch_size = 0

        self.is_updated = True
        self.init_weight_update = True
        self.weight = torch.nn.Parameter(
            torch.normal(0, 0.01, (self.num_local, embedding_size)).to(device)
        )

        if not isinstance(margin_loss, Callable):
            raise TypeError("margin_loss must be callable")
        self.margin_softmax = margin_loss

        # 选择 all_gather 函数
        if self.world_size > 1 and device == 'cuda':
            self.allgather = AllGatherFunc.apply
        else:
            self.allgather = local_all_gather

    def sample(self, labels, index_positive):
        with torch.no_grad():
            positive = torch.unique(labels[index_positive], sorted=True)
            if self.num_sample - positive.size(0) >= 0:
                perm = torch.rand(size=[self.num_local], device=self.device)
                perm[positive] = 2.0
                index = torch.topk(perm, k=self.num_sample)[1]
                index = index.sort()[0]
            else:
                index = positive
            self.weight_index = index
            labels[index_positive] = torch.searchsorted(index, labels[index_positive])
        return self.weight[self.weight_index]

    def forward(self, local_embeddings, local_labels):
        local_labels = local_labels.squeeze().long()
        batch_size = local_embeddings.size(0)

        # if self.last_batch_size == 0:
        self.last_batch_size = batch_size
        assert self.last_batch_size == batch_size, "batch size mismatch"

        # 准备 gather 列表
        _gather_embeddings = [
            torch.zeros((batch_size, self.embedding_size), device=self.device)
            for _ in range(self.world_size)
        ]
        _gather_labels = [
            torch.zeros(batch_size, dtype=torch.long, device=self.device)
            for _ in range(self.world_size)
        ]

        # AllGather embeddings
        _list_embeddings = self.allgather(local_embeddings, *_gather_embeddings)

        # AllGather labels（条件分支）
        if self.world_size > 1:
            dist.all_gather(_gather_labels, local_labels)
        else:
            _gather_labels[0] = local_labels

        embeddings = torch.cat(_list_embeddings)
        labels = torch.cat(_gather_labels).view(-1, 1)

        # 映射到本地类范围
        index_positive = (self.class_start <= labels) & (labels < self.class_start + self.num_local)
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        if self.sample_rate < 1:
            weight = self.sample(labels, index_positive)
        else:
            weight = self.weight

        # 前向计算（处理混合精度和 CPU）
        if self.device == 'cuda' and self.fp16:
            with torch.cuda.amp.autocast():
                norm_embeddings = normalize(embeddings)
                norm_weight_activated = normalize(weight)
                logits = linear(norm_embeddings, norm_weight_activated)
        else:
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)

        if self.fp16:
            logits = logits.float()

        logits = logits.clamp(-1, 1)
        logits = self.margin_softmax(logits, labels)
        loss = self.dist_cross_entropy(logits, labels)
        return  logits, loss


class DistCrossEntropyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, label, world_size):
        batch_size = logits.size(0)
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)

        # 条件 all_reduce
        if world_size > 1:
            dist.all_reduce(max_logits, dist.ReduceOp.MAX)
        logits.sub_(max_logits)
        logits.exp_()
        sum_logits_exp = torch.sum(logits, dim=1, keepdim=True)
        if world_size > 1:
            dist.all_reduce(sum_logits_exp, dist.ReduceOp.SUM)
        logits.div_(sum_logits_exp)

        index = torch.where(label != -1)[0]
        loss = torch.zeros(batch_size, 1, device=logits.device)
        loss[index] = logits[index].gather(1, label[index])
        if world_size > 1:
            dist.all_reduce(loss, dist.ReduceOp.SUM)

        ctx.save_for_backward(index, logits, label)
        ctx.world_size = world_size
        return loss.clamp_min_(1e-30).log_().mean() * (-1)

    @staticmethod
    def backward(ctx, loss_gradient):
        index, logits, label = ctx.saved_tensors
        batch_size = logits.size(0)
        one_hot = torch.zeros(size=[index.size(0), logits.size(1)], device=logits.device)
        one_hot.scatter_(1, label[index], 1)
        logits[index] -= one_hot
        logits.div_(batch_size)
        if ctx.world_size > 1:
            # 梯度同步（配合 AllGather 的反向）
            logits *= ctx.world_size
        return logits * loss_gradient.item(), None, None


class DistCrossEntropy(torch.nn.Module):
    def __init__(self, world_size=1):
        super().__init__()
        self.world_size = world_size

    def forward(self, logit_part, label_part):
        return DistCrossEntropyFunc.apply(logit_part, label_part, self.world_size)


class AllGatherFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, *gather_list):
        gather_list = list(gather_list)
        dist.all_gather(gather_list, tensor)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        grad_list = list(grads)
        rank = dist.get_rank()
        grad_out = grad_list[rank]

        dist_ops = [
            dist.reduce(grad_out, rank, dist.ReduceOp.SUM, async_op=True)
            if i == rank
            else dist.reduce(grad_list[i], i, dist.ReduceOp.SUM, async_op=True)
            for i in range(dist.get_world_size())
        ]
        for _op in dist_ops:
            _op.wait()
        grad_out *= len(grad_list)
        return (grad_out, *[None for _ in range(len(grad_list))])


def local_all_gather(tensor, *gather_list):
    return (tensor,)