import numbers
import os
import queue as Queue
import threading
from typing import Iterable

import mxnet as mx
import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils.show_data import denormalize_and_show
# from show_data import denormalize_and_show
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn, DistributedSampler
# from utils_distributed_sampler import get_dist_info, worker_init_fn, DistributedSampler


def get_dataloader(
        rec_path,
        idx_path,
        local_rank,
        batch_size,
        dali=True,
        dali_aug=True,
        seed=2048,
        num_workers=2,
        device='cpu',
        root_dir=None
) -> Iterable:
    rec = rec_path
    idx = idx_path
    train_set = None

    # Synthetic
    if root_dir:
        if root_dir == "synthetic":
            train_set = SyntheticDataset()
            dali = False

        # Image Folder
        else:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            train_set = ImageFolder(root_dir, transform)

    # DALI
    if dali:
        return dali_data_iter(
            batch_size=batch_size, rec_file=rec, idx_file=idx,
            num_threads=2, local_rank=local_rank, dali_aug=dali_aug, device=device)

    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class SyntheticDataset(Dataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000


def dali_data_iter(
        batch_size: int, rec_file: str, idx_file: str, num_threads: int,
        initial_fill=32768, random_shuffle=True,
        prefetch_queue_depth=1, local_rank=0, name="reader",
        mean=(127.5, 127.5, 127.5),
        std=(127.499214, 127.499214, 127.499214),
        dali_aug=False,
        device='cpu'
):
    """
    Parameters:
    ----------
    initial_fill: int
        Size of the buffer that is used for shuffling. If random_shuffle is False, this parameter is ignored.

    """
    # rank: int = distributed.get_rank()
    rank = 0
    # world_size: int = distributed.get_world_size()
    world_size = 1
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

    def dali_random_resize(img, resize_size, image_size=112):
        img = fn.resize(img, resize_x=resize_size, resize_y=resize_size)
        img = fn.resize(img, size=(image_size, image_size))
        return img

    def dali_random_gaussian_blur(img, window_size):
        img = fn.gaussian_blur(img, window_size=window_size * 2 + 1)
        return img

    def dali_random_gray(img, prob_gray):
        saturate = fn.random.coin_flip(probability=1 - prob_gray)
        saturate = fn.cast(saturate, dtype=types.FLOAT)
        img = fn.hsv(img, saturation=saturate)
        return img

    def dali_random_hsv(img, hue, saturation):
        img = fn.hsv(img, hue=hue, saturation=saturation)
        return img

    def multiplexing(condition, true_case, false_case):
        neg_condition = condition ^ True
        return condition * true_case + neg_condition * false_case

    condition_resize = fn.random.coin_flip(probability=0.1)
    size_resize = fn.random.uniform(range=(int(112 * 0.5), int(112 * 0.8)), dtype=types.FLOAT)
    condition_blur = fn.random.coin_flip(probability=0.2)
    window_size_blur = fn.random.uniform(range=(1, 2), dtype=types.INT32)
    condition_flip = fn.random.coin_flip(probability=0.5)
    condition_hsv = fn.random.coin_flip(probability=0.2)
    hsv_hue = fn.random.uniform(range=(0., 20.), dtype=types.FLOAT)
    hsv_saturation = fn.random.uniform(range=(1., 1.2), dtype=types.FLOAT)

    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads,
        device_id=local_rank, prefetch_queue_depth=prefetch_queue_depth, )
    condition_flip = fn.random.coin_flip(probability=0.5)
    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file, index_path=idx_file, initial_fill=initial_fill,
            num_shards=world_size, shard_id=rank,
            random_shuffle=random_shuffle, pad_last_batch=False, name=name)
        images = fn.decoders.image(jpegs, device=device, output_type=types.BGR)
        if dali_aug:
            # 原始
            images = fn.cast(images, dtype=types.UINT8)
            images = multiplexing(condition_resize, dali_random_resize(images, size_resize, image_size=112), images)
            images = multiplexing(condition_blur, dali_random_gaussian_blur(images, window_size_blur), images)
            images = multiplexing(condition_hsv, dali_random_hsv(images, hsv_hue, hsv_saturation), images)
            images = dali_random_gray(images, 0.1)
            # 新增
            # 随机亮度对比度
            condition_bc = fn.random.coin_flip(probability=0.3)
            images = multiplexing(condition_bc,
                                  fn.brightness_contrast(images,
                                                         brightness=fn.random.uniform(range=(0.8, 1.2),
                                                                                      dtype=types.FLOAT),
                                                         contrast=fn.random.uniform(range=(0.8, 1.2),
                                                                                    dtype=types.FLOAT)), images)

            # 3. 随机噪声（轻度高斯噪声，标准差 ≤ 10）
            condition_noise = fn.random.coin_flip(probability=0.15)
            images = multiplexing(condition_noise,
                                  fn.noise.gaussian(images, stddev=fn.random.uniform(range=(0, 10), dtype=types.FLOAT)),
                                  images)


            # 5. 随机 JPEG 压缩伪影
            condition_jpeg = fn.random.coin_flip(probability=0.1)
            quality_float = fn.random.uniform(range=(70, 95), dtype=types.FLOAT)
            quality = fn.cast(quality_float, dtype=types.INT32)
            images = fn.cast(images, dtype=types.UINT8)
            images = multiplexing(condition_jpeg,
                                  fn.jpeg_compression_distortion(images, quality=quality),
                                  images)

        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, mean=mean, std=std, mirror=condition_flip)
        pipe.set_outputs(images, labels)
    pipe.build()
    return DALIWarper(DALIClassificationIterator(pipelines=[pipe], reader_name=name, ), device)


@torch.no_grad()
class DALIWarper(object):
    def __init__(self, dali_iter, device):
        self.iter = dali_iter
        self.device = device
    def __next__(self):
        data_dict = self.iter.__next__()[0]
        # tensor_data = data_dict['data'].cuda()
        tensor_data = data_dict['data'].to(self.device)
        # tensor_label: torch.Tensor = data_dict['label'].cuda().long()
        tensor_label: torch.Tensor = data_dict['label'].to(self.device).long()
        tensor_label.squeeze_()
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def reset(self):
        self.iter.reset()

    def __len__(self):
        """返回当前迭代器剩余批次总数，通常等于一个 epoch 的总批次数（当尚未开始迭代时）"""
        return len(self.iter)


if __name__ == '__main__':

    rec_path = "/mnt/c/Users/Admin/Desktop/arcface-pytorch-main/datasets/train.rec"  # 特殊值
    idx_path = '/mnt/c/Users/Admin/Desktop/arcface-pytorch-main/datasets/train.idx'
    local_rank = 0  # 单GPU训练
    batch_size = 4

    dataloader = get_dataloader(
        rec_path,
        idx_path,
        local_rank,
        batch_size
    )

    for i, (imgs, labels) in enumerate(dataloader):
        print(f"Iteration {i + 1}:")
        print(
            f"  Image batch shape: {imgs.shape}")  # e.g., [32, 3, 224, 224] or [32, 3, 112, 112] depending on your data
        print(f"  Label batch shape: {labels.shape}")  # e.g., [32]
        print(f"  Image tensor device: {imgs.device}")
        print(f"  Label tensor device: {labels.device}")
        print(f"  Image tensor dtype: {imgs.dtype}")
        print(f"  Label tensor dtype: {labels.dtype}")

        denormalize_and_show(imgs, labels, to_uint8=True)
