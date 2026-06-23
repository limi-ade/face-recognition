import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch


def denormalize_and_show(images, labels, mean=(0.5, 0.5, 0.5), std=(0.502, 0.502, 0.502), max_images=4, to_uint8=False):
    if images.is_cuda:
        images = images.cpu()
    images = images.float()
    
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    images = images * std_t + mean_t   # 现在 [0,1]
    images = torch.clamp(images, 0, 1)
    
    images_np = images.numpy().transpose(0, 2, 3, 1)  # (N, H, W, C)
    
    if to_uint8:
        images_np = (images_np * 255).astype(np.uint8)
    
    num = min(len(images_np), max_images)
    fig, axes = plt.subplots(1, num, figsize=(4 * num, 4))
    if num == 1:
        axes = [axes]
    for i in range(num):
        axes[i].imshow(images_np[i])
        # axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].set_title(labels)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()