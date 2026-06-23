import random

import cv2
import numpy as np
import torch
from PIL import Image

import torch.nn as nn

def preprocess_input(image):

    image = (image - 127.5) / 128.0
    return image

# ---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
# ---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

    # ---------------------------------------------------#


#   对输入图像进行resize
# ---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    """
    支持 PIL Image 或 numpy 数组输入。
    :param image:            PIL Image 或 numpy 数组 (H,W,C)  BGR或RGB均可
    :param size:             目标尺寸 (width, height)
    :param letterbox_image:  是否保持宽高比并填充灰边
    :return:                 如果是PIL输入则返回PIL Image，否则返回numpy数组
    """
    # 记录输入类型
    was_pil = isinstance(image, Image.Image)
    if was_pil:
        # 转换为 numpy 数组 (RGB)
        img_np = np.array(image)
        # 注意：PIL 是 RGB，而 cv2 默认 BGR，这里不自动转换，因为后续 cv2.resize 对颜色顺序不敏感
        # 但如果后续有其他颜色操作，可能需要转换，此处保持 RGB。
    else:
        img_np = image

    ih, iw, c = img_np.shape
    w, h = size

    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        resized = cv2.resize(img_np, (nw, nh), interpolation=cv2.INTER_CUBIC)
        new_image = np.full((h, w, c), 128, dtype=np.uint8)
        left = (w - nw) // 2
        top = (h - nh) // 2
        new_image[top:top+nh, left:left+nw] = resized
        result = new_image
    else:
        result = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_CUBIC)

    if was_pil:

        return Image.fromarray(result)
    else:
        return result
    
def count_unique_ids(lst_file):

    unique_ids = set()
    with open(lst_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:

                label = float(parts[1])
                if label.is_integer():
                    label = int(label)
                unique_ids.add(label)

    return len(unique_ids)


# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# ---------------------------------------------------#
#   设置种子
# ---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------#
#   设置Dataloader的种子
# ---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# def preprocess_input(image):
#     image /= 255.0
#     image -= 0.5
#     image /= 0.5
#     return image



def preprocess_for_model(image):
    """
    根据配置文件预处理图像:
      mean = [127.5, 127.5, 127.5]
      norm = [0.0078125, 0.0078125, 0.0078125]
      输出范围约 [-0.996, 0.996]
    :param image: np.uint8, 形状 (H, W, C), 值域 [0, 255], 通道顺序 RGB 或 BGR (与模型要求一致)
    :return: np.float32, 形状 (1, C, H, W) 或 (C, H, W), 值域约 [-1, 1)
    """
    # 转换为 float32
    image = image.astype(np.float32)
    # 减均值, 乘归一化系数 (等效于 (image - mean) * norm)
    image = (image - 127.5) * 0.0078125
    # 将通道维度移到第0维 (若需要 NCHW，且无 batch 维)
    image = np.transpose(image, (2, 0, 1))   # (C, H, W)
    # 若需要添加 batch 维度，取消下一行的注释
    # image = np.expand_dims(image, axis=0)
    return image


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


def align_face_efficient(image, landmarks, face=None, out_size=112):
    # # 1. 先扩大裁剪人脸区域（加一点边距）
    # x1, y1, x2, y2 = face.location
    # margin = int((x2-x1) * 0.2)
    # x1 = max(0, x1 - margin)
    # y1 = max(0, y1 - margin)
    # x2 = min(image.shape[1], x2 + margin)
    # y2 = min(image.shape[0], y2 + margin)

    # # 2. 裁剪人脸区域
    # face_roi = image[y1:y2, x1:x2]

    # # 3. 调整关键点坐标到裁剪后的坐标系
    # adjusted_landmarks = landmarks - [x1, y1]

    # 4. 对裁剪后的小图进行对齐（而不是整张大图）
    # aligned, _ = align_face_by_106points(face_roi, adjusted_landmarks, out_size)
    aligned, _ = align_face_by_106points(image, landmarks, out_size)

    return aligned


def align_face_by_106points(image: np.ndarray, dense_landmarks: np.ndarray, out_size: int = 112) -> np.ndarray:
    """
    使用106点密集关键点进行人脸矫正
    
    Args:
        image: 原始BGR图像
        dense_landmarks: (106, 2) 的关键点数组
        out_size: 输出图像尺寸
    
    Returns:
        矫正后的人脸图像 (out_size, out_size, 3)
    """

    # 方法2：通过区域平均提高稳定性（推荐）
    if len(dense_landmarks) == 106:
        # 左眼区域平均（索引51-58）
        left_eye_indices = list(range(51, 59))
        left_eye = np.mean(dense_landmarks[left_eye_indices], axis=0)

        # 右眼区域平均（索引59-66）
        right_eye_indices = list(range(59, 67))
        right_eye = np.mean(dense_landmarks[right_eye_indices], axis=0)

        # 鼻尖
        nose_tip = dense_landmarks[100]

        # 嘴角
        left_mouth = dense_landmarks[77]
        right_mouth = dense_landmarks[71]

        selected_pts = np.array([
            left_eye, right_eye, nose_tip, left_mouth, right_mouth
        ], dtype=np.float32)

    else:

        selected_pts = dense_landmarks.astype(np.float32)

    # ArcFace 标准模板 (112x112)
    reference = np.array([
        [38.2946, 51.6963],  # 左眼
        [73.5318, 51.5014],  # 右眼
        [56.0252, 71.7366],  # 鼻尖
        [41.5493, 92.3655],  # 左嘴角
        [70.7299, 92.2041]  # 右嘴角
    ], dtype=np.float32)

    # 缩放到目标尺寸
    scale = out_size / 112.0
    reference *= scale

    # 估计仿射变换
    transform, inliers = cv2.estimateAffinePartial2D(
        selected_pts,
        reference,
        method=cv2.LMEDS
    )

    if transform is None:
        raise RuntimeError("仿射变换估计失败")

    # 应用变换
    aligned = cv2.warpAffine(image, transform, (out_size, out_size), flags=cv2.INTER_LINEAR)

    return aligned, transform


def cult_face(img, pks):
    h, w, c = img.shape
    x1, y1, x2, y2 = pks
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    im = img[y1:y2, x1:x2]
    return im


def get_largest_face(session, image):
    """从检测结果中返回面积最大的人脸，若无人脸则返回 None"""
    results = session.face_detection(image)
    if not results:
        return None
    # 假设每个人脸对象有 bbox 属性，格式为 (x1, y1, x2, y2)
    # 计算面积 = (x2 - x1) * (y2 - y1)
    largest_face = max(results, key=lambda face: (face.location[2] - face.location[0]) * (face.location[3] - face.location[1]))
    # return largest_face
    return results[0]




def print_model_grad(model):
    total_params = 0
    trainable_params_count = 0
    for name, param in model.named_parameters():
        total_params += 1
        if param.requires_grad:
            trainable_params_count += 1
            print(f"Trainable: {name}")  # 查看哪些参数可训练
        else:
            print(f"Frozen: {name}")     # 查看哪些参数被冻结
    print(f"Total params: {total_params}, Trainable params: {trainable_params_count}")


def fix_frozen_bn(m):
    for module in m.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
            module.eval()


def freeze_backbone_except_last_layers(model, unfreeze_layers, local_rank=0):
    # 首先冻结所有 backbone 参数
    for param in model.backbone.parameters():
        param.requires_grad = False

    # 解冻指定的顶层模块（匹配参数名前缀）
    for name, param in model.backbone.named_parameters():
        # 检查参数名是否以某个解冻模块名开头（如 'sep.', 'prelu.' 或直接等于 'prelu'）
        if any(name == layer or name.startswith(layer + '.') for layer in unfreeze_layers):
            param.requires_grad = True
            if local_rank == 0:
                print(f"Unfreeze layer: {name}")



def load_model_dict(model_config, pretrain, model):
    if model_config in ['mobilenetv1']: #  mobilenetv1 mbf
        model_dict = model.state_dict()
        pretrained_dict = torch.load(pretrain, map_location='cpu')
        # 重命名键：将开头 "arcface" 替换为 "backbone"
        new_pretrained_dict = {}
        for key, value in pretrained_dict.items():
            if key.startswith('arcface'):
                new_key = key.replace('arcface', 'backbone', 1)  # 只替换第一次出现
                new_pretrained_dict[new_key] = value
            else:
                new_pretrained_dict[key] = value

        # 过滤出模型所需的键（可选，去掉大小不匹配的层）
        # filtered_dict = {k: v for k, v in new_pretrained_dict.items() 
        #                 if k in model_dict and v.shape == model_dict[k].shape}

        model_dict.update(new_pretrained_dict)
        model.load_state_dict(model_dict, strict=True)
    else: # r18 ... ...
        model_dict = model.state_dict()
        pretrained_dict = torch.load(pretrain, map_location='cpu')

        # 在键名前加上 'backbone.'
        pretrained_dict = {f'backbone.{k}': v for k, v in pretrained_dict.items()}

        # 过滤形状匹配的层
        # filtered_dict = {k: v for k, v in pretrained_dict.items() 
        #                 if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=True)


# ============================================================================
# ArcFace 标准模板点 (112x112)
# ============================================================================
REFERENCE_POINTS = np.array([
    [38.2946, 51.6963],  # 左眼
    [73.5318, 51.5014],  # 右眼
    [56.0252, 71.7366],  # 鼻尖
    [41.5493, 92.3655],  # 左嘴角
    [70.7299, 92.2041]   # 右嘴角
], dtype=np.float32)

FACE_CROP_SIZE = 112


# ============================================================================
# Umeyama 相似性变换 v3 (RMS缩放)
# ============================================================================
def umeyama_similarity_transform_v3(dest, src):
    """
    Umeyama 相似性变换 - 使用 RMS 缩放
    
    参数顺序: dest (目标点集/模板) 在前, src (源点集/检测到的关键点) 在后
    返回: 3x3 仿射变换矩阵
    """
    n = src.shape[0]

    # 计算均值
    src_mean = np.mean(src, axis=0)
    dest_mean = np.mean(dest, axis=0)

    # 去均值
    src_centered = src - src_mean
    dest_centered = dest - dest_mean

    # 计算协方差矩阵
    H = src_centered.T @ dest_centered

    # SVD 分解
    U, S, Vt = np.linalg.svd(H)

    # 计算旋转矩阵
    R = Vt.T @ U.T

    # 处理镜像
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 计算缩放因子 (使用均方根)
    var_src = np.sum(src_centered ** 2) / n
    var_dest = np.sum(dest_centered ** 2) / n
    scale = np.sqrt(var_dest / var_src)

    # 计算平移向量
    t = dest_mean - scale * R @ src_mean

    # 构造 3x3 仿射变换矩阵
    transform = np.eye(3)
    transform[:2, :2] = scale * R
    transform[:2, 2] = t

    return transform


# ============================================================================
# 仿射变换 (使用 cv2.warpAffine)
# ============================================================================
def mnn_image_affine(image, transform_matrix, output_size=(112, 112)):
    """
    使用 cv2.warpAffine 进行仿射变换
    
    与 C++ ExecuteImageAffineProcessing 一致:
        - filterType: BILINEAR (cv2.INTER_LINEAR)
        - wrap: ZERO (cv2.BORDER_CONSTANT)
        - sourceFormat: BGR
        - destFormat: BGR
    """
    dst_w, dst_h = output_size
    
    # 提取 2x3 变换矩阵
    affine_matrix = transform_matrix[:2, :].astype(np.float32)
    
    # 执行仿射变换
    result = cv2.warpAffine(image, affine_matrix, (dst_w, dst_h), 
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    return result
