import os
import random

import inspireface as isf
import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import cv2
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import math


from utils.show_data import denormalize_and_show


from .utils import REFERENCE_POINTS, align_face_efficient, cult_face, cvtColor, mnn_image_affine, preprocess_for_model, resize_image, get_largest_face, \
    preprocess_input, umeyama_similarity_transform_v3
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# from utils.utils import align_face_efficient, cult_face, cvtColor, preprocess_for_model, resize_image, get_largest_face,preprocess_input

# Direct use path load inspireface model
isf.reload(model_name=None, resource_path='Pikachu_ori')
opt = isf.HF_ENABLE_FACE_RECOGNITION
session = isf.InspireFaceSession(opt, isf.HF_DETECT_MODE_ALWAYS_DETECT)

def preprocess_image(image_bgr, image_size=112):
    """
    使用 Umeyama v3 (RMS缩放) + cv2.warpAffine 进行预处理
    输入格式: RGB (与 PT/ONNX/MNN 训练时一致)
    """
    # # cv2.imread 返回 BGR
    # image_bgr = cv2.imread(image_path)  # BGR
    
    # 使用 BGR 图像进行人脸检测和关键点提取
    faces = get_largest_face(session, image_bgr)
    if faces is not None:
        # 获取5个关键点
        five_points = session.get_face_five_key_points(faces)
        five_points_np = np.array(five_points, dtype=np.float32)
        
        # 使用 Umeyama v3 (RMS缩放) 计算变换矩阵
        transform = umeyama_similarity_transform_v3(REFERENCE_POINTS, five_points_np)
        
        # 对 BGR 图像执行仿射变换
        image_bgr_aligned = mnn_image_affine(image_bgr, transform, output_size=(image_size, image_size))
        
        # BGR 转换为 RGB (与 PT/ONNX/MNN 训练时一致)
        image = cv2.cvtColor(image_bgr_aligned, cv2.COLOR_BGR2RGB)
    else:
        # image_rgb = cv2.imread(image_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    
    # 归一化: (img - 127.5) * 0.0078125
    img = (image.astype(np.float32) - 127.5) * 0.00784375
    
    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    
    # # 增加 batch 维度
    # img = np.expand_dims(img, axis=0)
    
    return img


def check_face(img):

    
    faces = get_largest_face(session, img) # 找到最大人脸
    if faces is None:
        return False
    else:
        return True


def read_lfw_pairs(pairs_path):
    pairs = []
    with open(pairs_path, 'r') as f:
        lines = f.readlines()
    # 第一行是 '10 300'，跳过
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 3:  # 正样本对： name idx1 idx2
            name, idx1, idx2 = parts
            pairs.append(('same', name, int(idx1), int(idx2)))
        elif len(parts) == 4:  # 负样本对： name1 idx1 name2 idx2
            name1, idx1, name2, idx2 = parts
            pairs.append(('diff', name1, int(idx1), name2, int(idx2)))
        else:
            # 忽略格式异常的行
            continue
    return pairs

def get_lfw_paths(dir, pairs_path):
    pairs = read_lfw_pairs(pairs_path)
    paths = []
    for pair in pairs:
        if pair[0] == 'same':
            _, name, idx1, idx2 = pair
            target_folder = os.path.join(dir, name)
            files = os.listdir(target_folder)
            file1 = files[idx1 - 1]
            file2 = files[idx2 - 1]

            # 构建两张图片的路径
            img1 = os.path.join(dir, name, file1)
            img2 = os.path.join(dir, name, file2)
            paths.append((img1, img2, 1))  # 1 表示相同身份
        else:  # diff
            _, name1, idx1, name2, idx2 = pair
            target_folder1 = os.path.join(dir, name1)
            target_folder2 = os.path.join(dir, name2)
            files1 = os.listdir(target_folder1)
            files2 = os.listdir(target_folder2)
            file1 = files1[idx1 - 1]
            file2 = files2[idx2 - 1]
            img1 = os.path.join(dir, name1, file1)
            img2 = os.path.join(dir, name2, file2)
            paths.append((img1, img2, 0))  # 0 表示不同身份
    return paths



class DistillDataset(Dataset):
    def __init__(self, input_shape, annotation_path, random=True, dali_aug=True):

        self.input_shape = input_shape   # (height, width)
        self.random = random
        self.dali_aug = dali_aug         # 是否启用与DALI相同的增强
        self.load_dataset_from_annotation(annotation_path)

    def __len__(self):
        return len(self.labels)

    def load_dataset_from_annotation(self,annotation_file_path):
        self.image_paths = []
        self.labels = []
        with open(annotation_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(';')
                label_str, image_path = parts
                label = int(label_str)
                self.image_paths.append(image_path)
                self.labels.append(label)



    def _random_resize(self, image):
        """随机缩放：先缩放到 [0.5*H, 0.8*H] 随机尺寸，再拉伸回 input_shape"""
        h, w = self.input_shape
        scale = np.random.uniform(0.5, 0.8)
        new_h = int(h * scale)
        new_w = int(w * scale)
        # 先缩放到随机尺寸
        image = image.resize((new_w, new_h), Image.BILINEAR)
        # 再拉伸回目标尺寸
        image = image.resize((w, h), Image.BILINEAR)
        return image

    def _random_gaussian_blur(self, image):
        """高斯模糊，核大小随机为3或5"""
        # PIL -> numpy (RGB)
        img_np = np.array(image)
        kernel_size = random.choice([3, 5])
        img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
        return Image.fromarray(img_np)

    def _random_hsv(self, image):
        img_np = np.array(image)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)

        hue_shift = np.random.uniform(0, 20)
        hsv[..., 0] += hue_shift
        hsv[..., 0] = np.clip(hsv[..., 0], 0, 180)

        sat_scale = np.random.uniform(1.0, 1.2)
        hsv[..., 1] *= sat_scale
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)

        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return Image.fromarray(rgb)

    def _random_gray(self, image, prob=0.1):

        if np.random.rand() < prob:
            image = image.convert('L').convert('RGB')
        return image

    # ---------- 主getitem ----------
    def __getitem__(self, index):

        img_path = self.image_paths[index]
        image = cvtColor(Image.open(img_path))

        # ========== 数据增强（与DALI对齐） ==========
        if self.dali_aug:
            # 1. 随机缩放 (概率0.1)
            if np.random.rand() < 0.1:
                image = self._random_resize(image)
            else:
                image = image.resize((self.input_shape[1], self.input_shape[0]), Image.BILINEAR)

            if np.random.rand() < 0.1:
                image = self._random_gaussian_blur(image)

            # 3. 随机HSV调整 (概率0.2)
            if np.random.rand() < 0.1:
                image = self._random_hsv(image)

            # 4. 随机灰度化 (概率0.1)
            image = self._random_gray(image, prob=0.1)

            # 5. 随机水平翻转 (概率0.5)
            if np.random.rand() < 0.5 and self.random:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            # 原有逻辑（无额外增强，仅简单翻转和letterbox resize）
            if np.random.rand() < .5 and self.random:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            # 使用原有的 letterbox resize
            image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image=True)

        image_RGB = np.array(image)  # RGB
        image_BGR = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2BGR) # BGR
        if not check_face(image_BGR):
            return None, None
        # image = preprocess_for_model(image_RGB)

        image = preprocess_image(image_BGR)
        
        # image = np.array(image, dtype='float32')
        # image = preprocess_input(image)      
        # image = np.transpose(image, (2, 0, 1))    # HWC -> CHW
        return image, image_BGR



def dataset_collate(batch):
    images = []
    targets = []
    for image, y in batch:
        images.append(image)
        targets.append(y)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    targets = torch.from_numpy(np.array(targets)).long()
    return images, targets


# def lfw_collate(batch):
#     """堆叠 LFW 图像对为 (N,C,H,W)，避免 batch_size=1 时缺少 batch 维。"""
#     data_a = torch.from_numpy(np.array([b[0] for b in batch ], dtype=np.float32))
#     data_p = torch.from_numpy(np.array([b[1] for b in batch], dtype=np.float32))
#     labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
#     return data_a, data_p, labels

def lfw_collate(batch):
    """堆叠 LFW 图像对为 (N,C,H,W)，并丢弃 batch 中的 None 元素。"""
    # 过滤掉 None 元素
    batch = [b for b in batch if b[0] is not None]
    # 如果过滤后 batch 为空，返回 None 表示跳过该 batch
    if len(batch) == 0:
        return None, None, None
    data_a = torch.from_numpy(np.array([b[0] for b in batch], dtype=np.float32))
    data_p = torch.from_numpy(np.array([b[1] for b in batch], dtype=np.float32))
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return data_a, data_p, labels

def distill_collate(batch):
    """堆叠 LFW 图像对为 (N,C,H,W)，并丢弃 batch 中的 None 元素。"""
    # 过滤掉 None 元素
    batch = [b for b in batch if b[0] is not None]
    # 如果过滤后 batch 为空，返回 None 表示跳过该 batch
    if len(batch) == 0:
        return None, None
    data_a = torch.from_numpy(np.array([b[0] for b in batch], dtype=np.float32))
    data_p = torch.from_numpy(np.array([b[1] for b in batch], dtype=np.float32))
    return data_a, data_p


def mobilefacenet_128_from_pp_prepro(image_path): 
    img = cv2.imread(image_path)
    # normalize to mean 0.5, std 0.5
    img = (img - 127.5) * 0.00784313725
    # BGR2RGB
    img = img[:, :, ::-1]
    img = img.transpose((2, 0, 1))
    # img = np.expand_dims(img, 0)
    img = img.astype('float32')
    return img

class LFWDataset(datasets.ImageFolder):
    def __init__(self, dir, pairs_path, image_size, transform=None):
        super(LFWDataset, self).__init__(dir, transform)
        self.image_size = image_size
        self.pairs_path = pairs_path
        self.validation_images = get_lfw_paths(dir, pairs_path)

    def __getitem__(self, index):
        path_1, path_2, issame = self.validation_images[index]


        image1, image2 = cv2.imread(path_1), cv2.imread(path_2) # BGR

        image1, image2 = preprocess_image(image1), preprocess_image(image2)

       
        return image1, image2, issame

    def __len__(self):
        return len(self.validation_images)


    
class MnnData(data.Dataset):
    def __init__(self, dir, pairs_path, image_size):
        super().__init__()
        self.image_size = image_size
        self.pairs_path = pairs_path
        self.validation_images = get_lfw_paths(dir, pairs_path)


    def __getitem__(self, index):
        path_1, path_2, issame = self.validation_images[index]
        
        image1_ori, image2_ori = cv2.imread(path_1), cv2.imread(path_2)  # BGR

        faces_1 = get_largest_face(session, image1_ori)
        faces_2 = get_largest_face(session, image2_ori)
        
        if faces_1 is None or faces_2 is None:
            return None, None, issame
        
        feature1 = session.face_feature_extract(image1_ori, faces_1)
        feature2 = session.face_feature_extract(image2_ori, faces_2)

        return feature1, feature2, issame

    def __len__(self):
        return len(self.validation_images)



class SemiSiameseDataset(Dataset):
    def __init__(self, imgs_dir_ori,annotation_path_ori, imgs_dir_en, input_shape=(112, 112),  random=True, dali_aug=True):
        self.input_shape = input_shape   # (height, width)
        self.random = random
        self.dali_aug = dali_aug
        self.imgs_dir_ori = imgs_dir_ori
        self.imgs_dir_en = imgs_dir_en
        self.image_paths = []

        self.load_dataset_from_annotation(annotation_path_ori)

    def __len__(self):
        return len(self.image_paths)

    def load_dataset_from_annotation(self, annotation_file_path):
        with open(annotation_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(';')
                # 无标签时，只用图像路径（忽略标签）
                if len(parts) >= 2:
                    _, image_path = parts
                else:
                    image_path = parts[0]
                self.image_paths.append(image_path)
        random.shuffle(self.image_paths)

    # ---------- 增强函数 ----------
    def _random_resize(self, image):
        h, w = self.input_shape
        scale = np.random.uniform(0.5, 0.8)
        new_h = int(h * scale)
        new_w = int(w * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        image = image.resize((w, h), Image.BILINEAR)
        return image

    def _random_gaussian_blur(self, image):
        img_np = np.array(image)
        kernel_size = random.choice([3, 5])
        img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
        return Image.fromarray(img_np)

    def _random_hsv(self, image):
        img_np = np.array(image)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
        hue_shift = np.random.uniform(-10, 10)
        hsv[..., 0] += hue_shift
        hsv[..., 0] = np.clip(hsv[..., 0], 0, 180)
        sat_scale = np.random.uniform(0.8, 1.2)
        hsv[..., 1] *= sat_scale
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return Image.fromarray(rgb)

    def _random_gray(self, image, prob=0.1):
        if np.random.rand() < prob:
            image = image.convert('L').convert('RGB')
        return image

    def _apply_augmentations(self, image, view_type='strong'):
        """
        view_type: 'strong' (view1) 或 'weak' (view2)
        不对称增强：strong 包含更多变化，weak 较保守
        """
        if not self.dali_aug:
            # 简单模式：仅翻转 + resize
            if self.random and np.random.rand() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = image.resize((self.input_shape[1], self.input_shape[0]), Image.BILINEAR)
            return image

        # --- DALI 风格增强 ---
        # 1. 随机缩放
        if view_type == 'strong':
            scale_prob = 0.2
        else:
            scale_prob = 0.05
        if np.random.rand() < scale_prob:
            image = self._random_resize(image)
        else:
            image = image.resize((self.input_shape[1], self.input_shape[0]), Image.BILINEAR)

        # 2. 高斯模糊（仅 strong）
        if view_type == 'strong' and np.random.rand() < 0.2:
            image = self._random_gaussian_blur(image)

        # 3. HSV 调整
        hsv_prob = 0.2 if view_type == 'strong' else 0.05
        if np.random.rand() < hsv_prob:
            image = self._random_hsv(image)

        # 4. 灰度化（strong 概率高）
        gray_prob = 0.1 if view_type == 'strong' else 0.02
        image = self._random_gray(image, prob=gray_prob)

        # 5. 水平翻转（独立）
        if self.random and np.random.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return image

    def _preprocess(self, image):
        image = np.array(image, dtype='float32')  # RGB
        image = preprocess_input(image)
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        return image

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img_path_ori = os.path.join(self.imgs_dir_ori, img_path)
        img_path_enhance = os.path.join(self.imgs_dir_en, img_path.split('.')[0] + '_v1.' + img_path.split('.')[1] )

        image_ori = cvtColor(Image.open(img_path_ori))  # RGB的PIL
        image_enhance = cvtColor(Image.open(img_path_enhance))

        if np.random.rand() < 0.9:
            # 生成两个不对称增强视图 view1
            view1 = self._apply_augmentations(image_ori.copy(), view_type='strong')
            view1 = self._preprocess(view1)

            # 生成两个不对称增强视图 view2
            view2 = self._apply_augmentations(image_ori.copy(), view_type='weak')
            view2 = self._preprocess(view2)

            return view1, view2  # 无标签
        else:
            # 生成两个不对称增强视图 view1
            view1 = self._preprocess(image_enhance)

            # 生成两个不对称增强视图 view2
            view2 = self._preprocess(image_ori)

            return view1, view2  # 无标签

class PairDataset(datasets.ImageFolder):
    def __init__(self, dir, pairs_path, image_size, transform=None):
        super(PairDataset, self).__init__(dir, transform)
        self.image_size = image_size
        self.pairs_path = pairs_path
        # self.noface = 0
        self.validation_images = get_lfw_paths(dir, pairs_path)

    def __getitem__(self, index):
        path_1, path_2, issame = self.validation_images[index]
 

        image1, image2 = cv2.imread(path_1), cv2.imread(path_2) # BGR

        faces_1, faces_2 = get_largest_face(session, image1), get_largest_face(session, image2)

        if faces_1 is not None and faces_2 is not None:


            image1, image2 = preprocess_image(image1), preprocess_image(image2)

            return image1, image2, issame
        
        else:
            return None, None, None
            


        

    def __len__(self):
        return len(self.validation_images)




if __name__ == '__main__':


    imgs_dir_ori = '/mnt/c/Users/Admin/Desktop/可用数据/CASIA-WebFace_蒸馏数据' 
    annotation_path_ori = '/mnt/c/Users/Admin/Desktop/可用数据/nomal_distill_train.txt'
    imgs_dir_en = '/mnt/c/Users/Admin/Desktop/可用数据/CASIA-WebFace_蒸馏数据_light' 
    dataset = SemiSiameseDataset( imgs_dir_ori,annotation_path_ori, imgs_dir_en)
    test_loader =DataLoader(dataset, batch_size=4,shuffle=False)
    for view1, view2 in test_loader:
        denormalize_and_show(view1, 'strong', to_uint8=True)
        denormalize_and_show(view2, 'weak', to_uint8=True)
    


