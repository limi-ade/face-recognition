import math
from functools import partial

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters,
                     warmup_iters_ratio=0.1, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.3, step_num=10,
                     cycles=1, restart_lr_ratio=0.5):
    """
    参数:
        cycles: 周期数，>1 时启用重启
        restart_lr_ratio: 每个新周期峰值 = 上周期峰值 * ratio (推荐0.5~0.8)
    每个周期内:
        1. 周期 warmup: 从 min_lr 线性升至当前周期 peak_lr (占周期长度的 warmup_ratio)
        2. 周期衰减: 按 cos/step 从 peak_lr 降至 min_lr
        3. 周期结尾 min_lr: 衰减结束后维持 min_lr 直到周期结束
    全局:
        - 前 warmup_total_iters 为全局 warmup (从 warmup_lr_start 升至 lr)
        - 最后 no_aug_iter 保持 min_lr
        - 中间每个周期的 peak_lr 依次递减
    """
    # 全局 warmup 设置
    warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
    warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
    no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)

    cycle_length = total_iters // cycles
    # 周期内部 warmup 长度 (占周期比例)
    cycle_warmup_ratio = warmup_iters_ratio
    cycle_warmup_length = max(int(cycle_length * cycle_warmup_ratio), 1)

    # 衰减函数
    if lr_decay_type == "cos":
        def decay_func(peak, min_lr, total_steps, iters):
            return min_lr + 0.5 * (peak - min_lr) * (1.0 + math.cos(math.pi * iters / total_steps))
    else:  # step
        def decay_func(peak, min_lr, total_steps, iters, step_num):
            if step_num <= 1:
                return peak
            decay_rate = (min_lr / peak) ** (1 / (step_num - 1))
            step_size = total_steps / step_num
            n = iters // step_size
            return peak * (decay_rate ** n)

    def scheduler(iters):
        # 1. 全局 warmup 阶段
        if iters < warmup_total_iters:
            return (lr - warmup_lr_start) * (iters / warmup_total_iters) ** 2 + warmup_lr_start

        # 2. 全局末尾 min_lr
        if iters >= total_iters - no_aug_iter:
            return min_lr

        # 3. 周期内计算
        cycle_idx = iters // cycle_length
        iters_in_cycle = iters % cycle_length

        # 当前周期峰值 (第一个周期峰值为 lr，之后衰减)
        current_peak = lr * (restart_lr_ratio ** cycle_idx)
        current_peak = max(current_peak, min_lr)   # 不低于 min_lr

        # 周期内的阶段划分
        # a) 周期 warmup (从 min_lr 到 current_peak)
        if iters_in_cycle < cycle_warmup_length:
            return min_lr + (current_peak - min_lr) * (iters_in_cycle / cycle_warmup_length)

        # b) 周期衰减阶段 (从 warmup 结束到周期结束)
        decay_iters = iters_in_cycle - cycle_warmup_length
        decay_total = cycle_length - cycle_warmup_length
        if decay_total <= 0:
            return current_peak
        if lr_decay_type == "cos":
            return decay_func(current_peak, min_lr, decay_total, decay_iters)
        else:
            return decay_func(current_peak, min_lr, decay_total, decay_iters, step_num)

    return scheduler

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
