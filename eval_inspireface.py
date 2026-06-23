import torch
from torch.utils.data import DataLoader

import inspireface as isf
from tqdm import tqdm

from utils.dataloader import LFWDataset, MnnData, lfw_collate
from utils.utils_metrics import test, test_mnn


def val_mnn(lfw_folder, pairs_path, png_save_path, log_interval=1, image_size=[112, 112]):
    dataset = MnnData(lfw_folder, pairs_path, image_size)
    test_loader = DataLoader(dataset, batch_size=1,shuffle=False, collate_fn=lfw_collate)
    test_mnn(test_loader, png_save_path, log_interval)



if __name__ == "__main__":
    
    png_save_path = "r18_l1.png"   
    lfw_folder = '/mnt/c/Users/Admin/Desktop/可用数据/train_data/eval/align'
    pairs_path = '/mnt/c/Users/Admin/Desktop/可用数据/train_data/eval/face_align_pair.txt'
    val_mnn(lfw_folder, pairs_path, png_save_path)
    png_save_path = "r18_l1_light.png"   
    lfw_folder = '/mnt/c/Users/Admin/Desktop/可用数据/train_data/eval/align_light'
    pairs_path = '/mnt/c/Users/Admin/Desktop/可用数据/train_data/eval/face_align_light_pair.txt'
    val_mnn(lfw_folder, pairs_path, png_save_path)
    
    