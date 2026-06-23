import os.path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import MNN
import torch
from scipy import interpolate
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.metrics import roc_curve


def evaluate(distances, labels, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, distances,
                                                        labels, nrof_folds=nrof_folds)
    thresholds = np.arange(0, 4, 0.001)

    val, val_std, far = calculate_val(thresholds, distances,
                                      labels, 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far, best_thresholds


def calculate_roc(thresholds, distances, labels, nrof_folds=10):
    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, distances[train_set], labels[train_set])

        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 distances[test_set],
                                                                                                 labels[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], distances[test_set],
                                                      labels[test_set])
        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, thresholds[best_threshold_index]


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, distances, labels, far_target=1e-3, nrof_folds=10):
    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # 计算当前 fold 下每个阈值的 FAR（基于训练集）
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, distances[train_set], labels[train_set])

        if np.max(far_train) >= far_target:
            # 使用局部变量，避免修改外部的 thresholds
            curr_far = np.array(far_train)
            curr_thresh = np.array(thresholds)

            # 过滤无效值
            valid = np.isfinite(curr_far) & np.isfinite(curr_thresh)
            curr_far = curr_far[valid]
            curr_thresh = curr_thresh[valid]

            # 按 far 排序
            sort_idx = np.argsort(curr_far)
            curr_far = curr_far[sort_idx]
            curr_thresh = curr_thresh[sort_idx]

            # 去重（保留第一个）
            unique_far, unique_idx = np.unique(curr_far, return_index=True)
            curr_thresh = curr_thresh[unique_idx]
            curr_far = unique_far

            if len(curr_far) >= 2:
                f = interpolate.interp1d(curr_far, curr_thresh, kind='slinear')
                threshold = f(far_target)
            else:
                # 数据点不足，取最小阈值
                threshold = curr_thresh[0] if len(curr_thresh) > 0 else 0.0
        else:
            # 所有阈值都无法达到 far_target，说明负样本距离很大（模型很好）
            # 使用负样本距离的最小值作为阈值，这样可以最大化 TAR 同时保持 FAR=0
            neg_dists = distances[train_set][labels[train_set] == 0]
            if len(neg_dists) > 0:
                threshold = np.min(neg_dists)
            else:
                threshold = thresholds[-1] if len(thresholds) > 0 else 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, distances[test_set], labels[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean

def calculate_val_fixed_far(thresholds, distances, labels, far_target=1e-3, nrof_folds=10):
    """
    固定 FAR 的评估，使用相似度分数（越大越可能是同人）绘制 ROC 并插值。
    参数 distances 为欧氏距离（越小越相似），内部自动转换为相似度分数。
    """
    nrof_pairs = len(labels)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    indices = np.arange(nrof_pairs)

    val_folds = []
    far_folds = []

    for train_set, test_set in k_fold.split(indices):
        # 关键修改：将欧氏距离转换为相似度分数（越大越同人）
        # 假设 out 已经 L2 归一化，距离范围 [0, 2]，取负即可
        scores = -distances[test_set]

        # 在测试集上计算 ROC 曲线（现在 score 越大越可能是正类）
        fpr, tpr, _ = roc_curve(
            labels[test_set],
            scores,
            pos_label=1,               # 1 表示同人
            drop_intermediate=False
        )

        # 去除 FPR 重复值
        unique_idx = np.unique(fpr, return_index=True)[1]
        fpr = fpr[unique_idx]
        tpr = tpr[unique_idx]

        # 处理边界
        if far_target < fpr[0]:
            tar_at_target = 0.0
        elif far_target > fpr[-1]:
            tar_at_target = tpr[-1]
        else:
            tar_at_target = np.interp(far_target, fpr, tpr)

        val_folds.append(tar_at_target)
        far_folds.append(far_target)   # 固定 FAR

    val_mean = np.mean(val_folds)
    val_std = np.std(val_folds)
    far_mean = far_target
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    if n_diff == 0:
        n_diff = 1
    if n_same == 0:
        return 0, 0
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def mnn_infer(interpreter, session, data):

    data = data.numpy()
    input_tensor = interpreter.getSessionInput(session)
    host_tensor = MNN.Tensor(data.shape, MNN.Halide_Type_Float, data, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(host_tensor)
    # 运行推理
    interpreter.runSession(session)
    # 获取输出
    output_tensor = interpreter.getSessionOutput(session)
    shape = output_tensor.getShape()

    # 创建主机端张量，并传入一个 numpy 数组作为存储空间
    host_output = MNN.Tensor(shape, output_tensor.getDataType(),
                             np.zeros(shape, dtype=np.float32),
                             MNN.Tensor_DimensionType_Caffe)

    # 将输出数据拷贝到 host_output 中
    output_tensor.copyToHostTensor(host_output)
    data_out = np.array(output_tensor.getData(), copy=False).reshape(shape).reshape(data.shape[0], -1)
    return data_out

def test(test_loader, model, png_save_path, log_interval, batch_size, cuda, interpreter=None):
    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        if isinstance(model, torch.nn.Module):
            with torch.no_grad():
                # --------------------------------------#
                #   加载数据，设置成cuda
                # --------------------------------------#
                data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
                if cuda:
                    data_a, data_p = data_a.cuda(), data_p.cuda()
                # --------------------------------------#
                #   传入模型预测，获得预测结果
                #   获得预测结果的距离
                # --------------------------------------#
                out_a, out_p = model(data_a), model(data_p)
                dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
                distances.append(dists.data.cpu().numpy())
        elif isinstance(model, MNN.Session):
            out_a = mnn_infer(interpreter, model, data_a)
            out_p = mnn_infer(interpreter, model, data_p)
            dists = np.sqrt(np.sum((out_a - out_p) ** 2, axis=1))
            distances.append(dists)

        labels.append(label.data.cpu().numpy())

        # --------------------------------------#
        #   打印
        # --------------------------------------#
        if batch_idx % log_interval == 0:
            pbar.set_description('Test Epoch: [{}/{} ({:.0f}%)]'.format(
                batch_idx * batch_size, len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))

    # --------------------------------------#
    #   转换成numpy
    # --------------------------------------#
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances, labels)
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Best_thresholds: %2.5f' % best_thresholds)
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    plot_roc(fpr, tpr, figure_name=png_save_path)


def test_mnn(test_loader, png_save_path, log_interval):
    labels, distances, cosine_sims = [], [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (out_a, out_p, label) in pbar:
        if out_a is None:
            continue
        out_a = out_a / out_a.norm(dim=1, keepdim=True)
        out_p = out_p / out_p.norm(dim=1, keepdim=True)

        # 计算欧氏距离
        dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
        distances.append(dists.data.cpu().numpy())

        # 计算余弦相似度
        cos_sims = torch.sum(out_a * out_p, dim=1)
        cosine_sims.append(cos_sims.data.cpu().numpy())

        labels.append(label.data.cpu().numpy())

        # --------------------------------------#
        #   打印
        # --------------------------------------#
        if batch_idx % log_interval == 0:
            pbar.set_description('Test Epoch: [{}/{} ({:.0f}%)]'.format(
                batch_idx, len(test_loader),
                100. * batch_idx / len(test_loader)))

    # --------------------------------------#
    #   转换成numpy
    # --------------------------------------#
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])
    cosine_sims = np.array([subsim for sim in cosine_sims for subsim in sim])

    # 打印距离分布信息
    print('=' * 60)
    print('距离分布统计:')
    print(f'  正样本对数量: {np.sum(labels == 1)}')
    print(f'  负样本对数量: {np.sum(labels == 0)}')
    print(f'  正样本对距离: min={distances[labels==1].min():.4f}, max={distances[labels==1].max():.4f}, mean={distances[labels==1].mean():.4f}')
    print(f'  负样本对距离: min={distances[labels==0].min():.4f}, max={distances[labels==0].max():.4f}, mean={distances[labels==0].mean():.4f}')
    print('=' * 60)

    # 打印余弦相似度分布信息
    print('=' * 60)
    print('余弦相似度分布统计:')
    print(f'  正样本对数量: {np.sum(labels == 1)}')
    print(f'  负样本对数量: {np.sum(labels == 0)}')
    print(f'  正样本对余弦相似度: min={cosine_sims[labels==1].min():.4f}, max={cosine_sims[labels==1].max():.4f}, mean={cosine_sims[labels==1].mean():.4f}')
    print(f'  负样本对余弦相似度: min={cosine_sims[labels==0].min():.4f}, max={cosine_sims[labels==0].max():.4f}, mean={cosine_sims[labels==0].mean():.4f}')
    print('=' * 60)

    tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances, labels)
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Best_thresholds: %2.5f' % best_thresholds)
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    plot_roc(fpr, tpr, figure_name=png_save_path)
    png_save_path_log = png_save_path.split('.')[0] + '_log.png'
    plot_roc_log(fpr, tpr, figure_name=png_save_path_log)

    # 可视化余弦相似度分布（传入最佳阈值）
    plot_cosine_similarity_distribution(cosine_sims, labels, png_save_path, best_thresholds)



def plot_roc(fpr, tpr, figure_name="roc.png"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc

    # 关键：按 FPR 排序，确保单调递增
    sorted_idx = np.argsort(fpr)
    fpr_sorted = np.array(fpr)[sorted_idx]
    tpr_sorted = np.array(tpr)[sorted_idx]

    roc_auc = auc(fpr_sorted, tpr_sorted)

    fig = plt.figure()
    lw = 2
    plt.plot(fpr_sorted, tpr_sorted, color='darkorange',
             lw=lw, label='ROC curve (area = %0.8f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(figure_name, dpi=fig.dpi)

def plot_roc_log(fpr, tpr, figure_name="roc.png", log_scale=True):
    """
    绘制 ROC 曲线，支持线性或对数刻度。
    Y 轴刻度以 0.05 为间隔，方便观察 TPR 的细微变化。
    """
    # 按 fpr 排序（必须确保单调递增）
    sorted_idx = np.argsort(fpr)
    fpr_sorted = np.array(fpr)[sorted_idx]
    tpr_sorted = np.array(tpr)[sorted_idx]

    roc_auc = auc(fpr_sorted, tpr_sorted)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_sorted, tpr_sorted, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.8f})')

    if log_scale:
        plt.xscale('log')
        plt.xlabel('False Positive Rate (log scale)')
        plt.xlim([1e-4, 1.0])
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
    else:
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.xlabel('False Positive Rate')

    # ========== 细化 Y 轴刻度 ==========
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.ylim([0.0, 1.05])
    # =================================

    plt.ylabel('True Positive Rate (Validation Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figure_name, dpi=150)
    plt.close()


def plot_cosine_similarity_distribution(cosine_sims, labels, png_save_path, best_threshold=None):
    """
    可视化正负样本对的余弦相似度分布，并标记最佳阈值

    Args:
        cosine_sims: 余弦相似度数组
        labels: 标签数组 (1=正样本, 0=负样本)
        png_save_path: ROC 曲线保存路径（用于生成余弦相似度分布图路径）
        best_threshold: 最佳阈值（欧氏距离，会转换为余弦相似度）
    """
    # 分离正负样本
    pos_sims = cosine_sims[labels == 1]
    neg_sims = cosine_sims[labels == 0]

    # 计算最佳余弦相似度阈值（从欧氏距离转换）
    # 欧氏距离 d = sqrt(2 * (1 - cos_sim)) => cos_sim = 1 - d^2 / 2
    if best_threshold is not None:
        best_cos_sim = 1 - (best_threshold ** 2) / 2
    else:
        best_cos_sim = None

    # 创建图形
    plt.figure(figsize=(14, 6))

    # 绘制直方图
    plt.subplot(1, 2, 1)
    plt.hist(pos_sims, bins=50, alpha=0.7, label='Positive (Same Person)', color='green', density=True)
    plt.hist(neg_sims, bins=50, alpha=0.7, label='Negative (Different Person)', color='red', density=True)

    # 标记最佳阈值
    if best_cos_sim is not None:
        plt.axvline(x=best_cos_sim, color='blue', linestyle='--', linewidth=2,
                   label=f'Best Threshold (Cosine Sim={best_cos_sim:.4f})')

    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Cosine Similarity Distribution (Histogram)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 绘制核密度估计
    plt.subplot(1, 2, 2)
    from scipy.stats import gaussian_kde

    # 计算核密度估计
    x_range = np.linspace(min(cosine_sims.min(), -1), max(cosine_sims.max(), 1), 500)
    kde_pos = None
    kde_neg = None

    if len(pos_sims) > 0:
        kde_pos = gaussian_kde(pos_sims)
        plt.plot(x_range, kde_pos(x_range), label='Positive (Same Person)', color='green', linewidth=2)

    if len(neg_sims) > 0:
        kde_neg = gaussian_kde(neg_sims)
        plt.plot(x_range, kde_neg(x_range), label='Negative (Different Person)', color='red', linewidth=2)

    # 标记最佳阈值
    if best_cos_sim is not None:
        plt.axvline(x=best_cos_sim, color='blue', linestyle='--', linewidth=2,
                   label=f'Best Threshold (Cosine Sim={best_cos_sim:.4f})')

        # 计算交点（如果两条曲线都存在）
        if kde_pos is not None and kde_neg is not None:
            # 找到两条曲线最接近的点
            diff = np.abs(kde_pos(x_range) - kde_neg(x_range))
            intersection_idx = np.argmin(diff)
            intersection_x = x_range[intersection_idx]
            plt.axvline(x=intersection_x, color='orange', linestyle=':', linewidth=2,
                       label=f'Distribution Intersection (Cosine Sim={intersection_x:.4f})')

    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Cosine Similarity Distribution (KDE)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    cos_sim_path = png_save_path.split('.')[0] + '_cosine_similarity.png'
    plt.savefig(cos_sim_path, dpi=150)
    plt.close()

    print(f'Cosine similarity distribution plot saved: {cos_sim_path}')

    # 打印阈值建议
    print('=' * 60)
    print('阈值建议 (基于余弦相似度):')
    if best_cos_sim is not None:
        print(f'最佳阈值(欧氏距离={best_threshold:.4f}): 余弦相似度={best_cos_sim:.4f}')

    # 计算交点阈值
    if kde_pos is not None and kde_neg is not None:
        diff = np.abs(kde_pos(x_range) - kde_neg(x_range))
        intersection_idx = np.argmin(diff)
        intersection_x = x_range[intersection_idx]
        print(f'分布交点阈值: 余弦相似度={intersection_x:.4f}')

    # 计算统计阈值
    if len(pos_sims) > 0 and len(neg_sims) > 0:
        # 中点阈值
        midpoint = (pos_sims.mean() + neg_sims.mean()) / 2
        print(f'中点阈值: 余弦相似度={midpoint:.4f}')

        # 正样本最大值和负样本最小值的中点
        boundary = (pos_sims.max() + neg_sims.min()) / 2
        print(f'边界中点阈值: 余弦相似度={boundary:.4f}')
    print('=' * 60)