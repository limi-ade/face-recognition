import os

from tqdm import tqdm



import os
import random
import glob

def generate_lfw_pairs(data_root, output_file, n_folds=10, n_pairs_per_fold=355):
    """
    生成类似 lfw_pair.txt 的 10折交叉验证文件
    
    Args:
        data_root (str): 数据集根目录，包含 ID 子文件夹（如 dataset/1/）
        output_file (str): 输出文件名
        n_folds (int): 交叉验证折数
        n_pairs_per_fold (int): 每折中包含的正样本对数量
    """
    
    # 1. 扫描数据集，获取每个人的所有图片路径
    # 使用 sorted 确保 ID 顺序一致（例如 1, 2, 3...）
    # 过滤掉非文件夹的文件
    all_ids = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    
    # 存储结构: person_images = {'ID': ['file1.jpg', 'file2.jpg', ...]}
    person_images = {}
    for pid in all_ids:
        person_dir = os.path.join(data_root, pid)
        # 查找该 ID 下的所有图片 (jpg, jpeg, png, bmp)
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(person_dir, ext)))
            # 某些系统不区分大小写，可能需要再次搜索 *.JPG 等
            image_files.extend(glob.glob(os.path.join(person_dir, ext.upper())))
        
        # 去除重复并排序，保证可复现性
        image_files = sorted(list(set(image_files)))
        
        # 只保留文件名，用于写入配对文件
        person_images[pid] = [os.path.basename(f) for f in image_files]
    
    # 2. 生成正样本对 (Same ID) 和 负样本对 (Different ID)
    # 正样本对格式: ID   index1   index2 (这里的 index 指的是文件名在排序列表中的索引)
    # 负样本对格式: ID1  index1  ID2  index2
    
    positive_pairs = []
    negative_pairs = []
    
    # 获取所有 ID 列表
    ids_list = list(person_images.keys())
    
    print(f"检测到 {len(ids_list)} 个 ID...")
    
    # 遍历每个 ID，尝试生成正负样本
    # 这里为了保证生成足够的 pair，我们对 ID 进行随机采样循环
    # 如果图片不够 2 张的 ID 会被跳过
    for _ in range(n_folds * n_pairs_per_fold * 5): # 乘 5 是预生成更多的样本，防止数量不足
        # 随机选择一个 ID
        curr_id = random.choice(ids_list)
        curr_images = person_images[curr_id]
        
        # 如果该人的图片少于 2 张，无法生成正样本，跳过
        if len(curr_images) < 2:
            continue
            
        # 1. 生成正样本对
        idx1, idx2 = random.sample(range(len(curr_images)), 2)
        positive_pairs.append(f"{curr_id}\t{idx1+1}\t{idx2+1}") # 索引从1开始写入文件
        
        # 2. 生成负样本对
        # 随机找一个不同的人
        other_id = random.choice([id for id in ids_list if id != curr_id])
        other_images = person_images[other_id]
        
        other_idx = random.randint(0, len(other_images) - 1)
        # 格式：ID1 index1 ID2 index2
        negative_pairs.append(f"{curr_id}\t{idx1+1}\t{other_id}\t{other_idx+1}")

    # 3. 修剪样本数量，确保每一折的数量都足够
    # 我们希望每个 fold 由 (n_pairs_per_fold) 个正样本 + (n_pairs_per_fold) 个负样本组成
    # 因此总的正样本数需要 >= n_folds * n_pairs_per_fold
    
    # 随机打乱并截取指定数量
    random.shuffle(positive_pairs)
    random.shuffle(negative_pairs)
    
    total_needed = n_folds * n_pairs_per_fold
    # 检查是否生成了足够的数据
    if len(positive_pairs) < total_needed or len(negative_pairs) < total_needed:
        print(f"警告：生成的数据量不足。需要 {total_needed} 对，实际正样本 {len(positive_pairs)} 对，负样本 {len(negative_pairs)} 对。")
        print("请确保 dataset 文件夹中有足够多的 ID 和图片数据（每个 ID 至少 2 张）。")
        return

    positive_pairs = positive_pairs[:total_needed]
    negative_pairs = negative_pairs[:total_needed]

    # 4. 写入到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入头部: 10 (折数)  300 (每折对数)
        f.write(f"{n_folds}\t{n_pairs_per_fold}\n")
        
        for fold in range(n_folds):
            # 获取当前折对应的正负样本 (切片)
            start_idx = fold * n_pairs_per_fold
            end_idx = (fold + 1) * n_pairs_per_fold
            
            fold_pos = positive_pairs[start_idx:end_idx]
            fold_neg = negative_pairs[start_idx:end_idx]
            
            # 写入正样本
            for line in fold_pos:
                f.write(line + "\n")
            # 写入负样本
            for line in fold_neg:
                f.write(line + "\n")
                
    print(f"成功生成配对文件: {output_file}")
    print(f"总共包含 {n_folds} 折，每折包含 {n_pairs_per_fold} 对正样本和 {n_pairs_per_fold} 对负样本。")


def scan_folder(path):
    
    files = os.listdir(path)
    paths = [os.path.join(path.split('/')[-1], f) for f in files]
    return paths

def scan_dir(path):

    msg = []
    folders = os.listdir(path)
    for f in folders:
        f_path = os.path.join(path, f)
        paths = scan_folder(f_path)
        msg.extend(paths)

    random.shuffle(msg)

    with open('semisiamese.txt', 'w') as f:
        for i in tqdm(msg):
            f.write(i + '\n')

if __name__ == '__main__':

    data_root_path = "distill_dataset_75_25/val" 
    output_file_name = "distill_dataset_75_25/val_pair.txt"
    
    generate_lfw_pairs(data_root_path, output_file_name, n_folds=10, n_pairs_per_fold=160)
