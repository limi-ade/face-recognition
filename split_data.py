# -*- coding: utf-8 -*-
import os
import random
import shutil
from tqdm import tqdm


def split_face_dataset(src_dir, dst_dir, train_ratio=0.8, val_ratio=0.2, seed=42):
    """
    按比例划分人脸识别数据集，并处理小样本情况。

    参数:
    src_dir: 原始数据集根目录
    dst_dir: 划分后数据集保存的根目录
    train_ratio: 训练集比例 (0-1)
    val_ratio: 验证集比例 (0-1)
    seed: 随机种子，保证结果可复现
    """

    # 1. 设置随机种子
    random.seed(seed)

    # 2. 获取所有类别文件夹
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

    if not classes:
        print("❌ 错误：源目录下未找到类别子文件夹。")
        return

    print(f"🔍 检测到 {len(classes)} 个类别，开始处理...")

    # 3. 遍历每个类别进行处理
    for class_name in tqdm(classes, desc="处理进度"):
        src_class_path = os.path.join(src_dir, class_name)

        # 目标路径
        dst_train_class_path = os.path.join(dst_dir, 'train', class_name)
        dst_val_class_path = os.path.join(dst_dir, 'val', class_name)

        # 创建目标文件夹
        os.makedirs(dst_train_class_path, exist_ok=True)
        os.makedirs(dst_val_class_path, exist_ok=True)

        # 获取该类别下的所有图片文件
        images = [f for f in os.listdir(src_class_path)
                  if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

        num_images = len(images)

        # --- 核心逻辑：处理小样本情况 ---
        if num_images < 2:
            # 如果图片少于2张，在train和val中各复制一份
            # 这样可以保证每个类别在两个集合中都有数据，防止某些框架报错
            for img in images:
                src_img_path = os.path.join(src_class_path, img)
                shutil.copy(src_img_path, os.path.join(dst_train_class_path, img))
                shutil.copy(src_img_path, os.path.join(dst_val_class_path, img))
            continue

        # --- 正常情况：按比例随机划分 ---
        # 打乱图片顺序
        random.shuffle(images)

        # 计算切分点
        train_count = int(num_images * train_ratio)

        # 确保验证集至少有1张图（防止比例极端导致验证集为空）
        if train_count == num_images and num_images > 1:
            train_count -= 1

        train_images = images[:train_count]
        val_images = images[train_count:]

        # 复制训练集图片
        for img in train_images:
            src_path = os.path.join(src_class_path, img)
            dst_path = os.path.join(dst_train_class_path, img)
            shutil.copy(src_path, dst_path)

        # 复制验证集图片
        for img in val_images:
            src_path = os.path.join(src_class_path, img)
            dst_path = os.path.join(dst_val_class_path, img)
            shutil.copy(src_path, dst_path)

    print(f"✅ 数据集划分完成！")
    print(f"📂 保存位置: {dst_dir}")


# ================= 配置区域 =================
if __name__ == '__main__':

    SOURCE_DIR = 'distill_dataset'
    TARGET_DIR = 'distill_dataset_75_25'

    TRAIN_RATIO = 0.75
    VAL_RATIO = 0.25

    # 执行划分
    split_face_dataset(SOURCE_DIR, TARGET_DIR, TRAIN_RATIO, VAL_RATIO)
    