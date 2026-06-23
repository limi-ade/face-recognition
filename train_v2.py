import os

import yaml
import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from easydict import EasyDict
from torch.utils.data import DataLoader
from functools import partial
from loss.losses import BatchHardTripletLoss
from nets.arcface import Arcface
# from nets.arcface_ori import ArcfaceOri
from nets.arcface_training import get_lr_scheduler, set_optimizer_lr
from nets.semisiamesestudent import SemiSiameseStudent
from utils.callback import LossHistory
from utils.dataloader import DistillDataset, LFWDataset, PairDataset, SemiSiameseDataset, dataset_collate, distill_collate, lfw_collate
from utils.dataset_train import get_dataloader
from utils.utils import count_unique_ids, fix_frozen_bn, freeze_backbone_except_last_layers, load_model_dict, seed_everything, show_config, worker_init_fn
from utils.utils_fit import distill, fit_one_epoch, fit_with_distill, fit_with_semi_siamese_v2, val_run
from utils.utils_metrics import plot_roc



def train(cfg_path, **kwargs):

    # config
    config = yaml.load(open(cfg_path, encoding='utf8'), Loader=yaml.FullLoader)
    config = EasyDict(config)  # convert to dict
    model_config = config.Arch
    train_config = config.Train_param
    global_config = config.Global
    dataset_config = config.Dataset
    seed_everything(global_config.seed)
    ngpus_per_node = torch.cuda.device_count()
    if global_config.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = global_config.device
        local_rank = 0
        rank = 0

    # model loader
    num_classes = count_unique_ids(global_config.annotation_path)
    model = Arcface(model_config.loss, model_config.margin_list, model_config.interclass_filtering_threshold,
                    model_config.embedding_size, num_classes, model_config.sample_rate, model_config.fp16,
                    model_config.backbone_s, model_config.pretrained, global_config.distributed, device, **kwargs)
    print('load model:')
    print(model_config.pretrained)
    if model_config.pretrained and not model_config.resume: 
        print('load from pretrained not resume')
        load_model_dict(model_config.backbone_s, model_config.pretrained, model)

    elif model_config.pretrained and model_config.resume:
        print('load from pretrained with resume')
        model_dict = torch.load(model_config.pretrained, map_location='cpu')
        backbone_dict = model_dict['state_dict_backbone']
        # fc_dict = model_dict['state_dict_softmax_fc']
        # backbone_dict = {f'backbone.{k}': v for k, v in backbone_dict.items()}
        model.backbone.load_state_dict(backbone_dict, strict=True)
        # model.module_partial_fc.load_state_dict(fc_dict, strict=True)
    
    if global_config.fine_tune:
        # freeze_backbone_except_last_layers(model, global_config.unfreeze_layers) 

        head_params = []    
        other_params = []  

        for name, param in model.backbone.named_parameters():
            if global_config.unfreeze_layers[0] in name:
                other_params.append(param)
                param.requires_grad = True
                if local_rank == 0:
                    print(f"Unfreeze layer: {name}")
            elif any(key in name for key in global_config.unfreeze_layers[1:]):
                head_params.append(param)
                param.requires_grad = True
                if local_rank == 0:
                    print(f"Unfreeze layer: {name}")
            else:
                param.requires_grad = False  # 其他层全部冻结
        trainable_params = None
        # trainable_params = [p for p in model.parameters() if p.requires_grad]  # 使用预训练权重from_pp
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    # dataloader
    train_loader = get_dataloader(
        dataset_config.train_rec,
        dataset_config.train_idx,
        local_rank,
        dataset_config.train_batch_size,
        dataset_config.dali,
        dataset_config.train_dali_aug,
        global_config.seed,
        dataset_config.num_workers,
        dataset_config.train_device
    )

    val_loader = get_dataloader(
        dataset_config.val_rec,
        dataset_config.val_idx,
        local_rank,
        dataset_config.val_batch_size,
        dataset_config.dali,
        dataset_config.val_dali_aug,
        global_config.seed,
        dataset_config.num_workers,
        dataset_config.val_device
        )
    
    imgs_dir_ori = dataset_config.semisiame_train_imgs_ori
    annotation_path_ori =  dataset_config.semisiame_train_annotation
    imgs_dir_en = dataset_config.semisiame_train_imgs_en
    semisiame_train_data = SemiSiameseDataset(imgs_dir_ori,annotation_path_ori, imgs_dir_en)  # 半孪生用什么数据集
    semisiame_workers = dataset_config.num_workers
    semisiame_train_loader = DataLoader(
        semisiame_train_data,
        batch_size=dataset_config.semisiame_train_batch_size,
        shuffle=dataset_config.semisiame_train_shuffle,
        num_workers=semisiame_workers,
        pin_memory=not global_config.device == 'cpu',
        drop_last=True,
    )

    distill_train_data = DistillDataset(dataset_config.input_shape[:-1], dataset_config.distill_annotation_path)
    distill_workers = dataset_config.num_workers
    distill_train_loader = DataLoader(
        distill_train_data,
        batch_size=dataset_config.distill_batch_size,
        shuffle=dataset_config.distill_shuffle,
        num_workers=distill_workers,
        pin_memory=not global_config.device == 'cpu',
        drop_last=True,
        collate_fn=distill_collate
    )

    datapair_val_data = PairDataset(dir=dataset_config.semisiame_val_path, pairs_path=dataset_config.semisiame_val_pairs_path,
                            image_size=dataset_config.input_shape)
    datapair_val_loader = DataLoader(
        datapair_val_data,
        batch_size=dataset_config.semisiame_val_batch_size,
        shuffle=dataset_config.semisiame_val_shuffle,
        num_workers=dataset_config.num_workers,
        collate_fn=lfw_collate,
    )

    # teacher model
    if not model_config.distill_model:
        # 初始化dataloader 加载 
        print('load teacher:  pikaqiu')
    else:
        model_teacher = Arcface(model_config.loss, model_config.margin_list, model_config.interclass_filtering_threshold,
                    512, num_classes, model_config.sample_rate, model_config.fp16,
                    model_config.backbone_t, model_config.pretrained, global_config.distributed, device, **kwargs)       
        print('load teacher: ')
        print(model_config.distill_model)
        load_model_dict(model_config.backbone_t, model_config.distill_model, model_teacher)
        for param in model_teacher.parameters():
            param.requires_grad = False

        model_teacher.to(device)
        model_teacher.eval()

    # trip loss
    triplet_loss_fn = BatchHardTripletLoss()

 
    if local_rank == 0:
        loss_history = LossHistory(train_config.save_dir, cfg_path, model, input_shape=dataset_config.input_shape)
    else:
        loss_history = None

    # cross-validation
    if global_config.eval_flag:

        lfw_data = LFWDataset(dir=dataset_config.lfw_dir_path, pairs_path=dataset_config.lfw_pairs_path,
                              image_size=dataset_config.input_shape)
        lfw_loader = DataLoader(
            lfw_data,
            batch_size=dataset_config.lfw_batch_size,
            shuffle=dataset_config.lfw_shuffle,
            num_workers=dataset_config.num_workers,
            collate_fn=lfw_collate,
        )


        face_val_data = PairDataset(dir=dataset_config.face_val_path, pairs_path=dataset_config.face_val_pairs_path,
                              image_size=dataset_config.input_shape)
        face_data_loader = DataLoader(
            face_val_data,
            batch_size=dataset_config.face_val_batch_size,
            shuffle=dataset_config.semisiame_val_shuffle,
            num_workers=dataset_config.num_workers,
            collate_fn=lfw_collate,
        )

        enface_val_data = PairDataset(dir=dataset_config.enface_val_path, pairs_path=dataset_config.enface_val_pairs_path,
                              image_size=dataset_config.input_shape)
        enface_data_loader = DataLoader(
            enface_val_data,
            batch_size=dataset_config.enface_val_batch_size,
            shuffle=dataset_config.enface_val_shuffle,
            num_workers=dataset_config.num_workers,
            collate_fn=lfw_collate,
        )


    # show config
    show_config(
        num_classes=num_classes, backbone=model_config.backbone_s, model_path=model_config.pretrained,
        input_shape=dataset_config.input_shape,
        Init_Epoch=train_config.init_Epoch, Epoch=train_config.epoch, batch_size=dataset_config.train_batch_size,
        Init_lr=train_config.init_lr, Min_lr=float(train_config.init_lr) * 0.01, optimizer_type=train_config.optimizer_type,
        momentum=train_config.momentum, lr_decay_type=train_config.lr_decay_type,
        save_period=train_config.save_period, save_dir=train_config.save_dir, num_workers=dataset_config.num_workers,
        distill_flag=global_config.distill_flag, semisiamese_distill_flag=global_config.semisiamese_distill_flag, cls_flag = global_config.cls_flag, eval_flag=global_config.eval_flag
    )

    nbs = 64
    lr_limit_max = 1e-3 if train_config.optimizer_type == 'adam' else 1e-1
    lr_limit_min = 3e-4 if train_config.optimizer_type == 'adam' else 5e-4
    init_lr_fit = min(max(float(dataset_config.train_batch_size) / nbs * float(train_config.init_lr), lr_limit_min), lr_limit_max)
    min_lr_fit = min(max(float(dataset_config.train_batch_size) / nbs * float(train_config.init_lr) * 0.01, lr_limit_min * 1e-2),
                     lr_limit_max * 1e-2)

    # different lr
    if not trainable_params:
        optimizer = {
        'adam': optim.Adam(
            [
                {'params': head_params, 'lr': init_lr_fit},
                {'params': other_params, 'lr': init_lr_fit * train_config.lr_rate}  
            ], 
            betas=(float(train_config.momentum), 0.999), 
            weight_decay=float(train_config.weight_decay)),
        'sgd': optim.SGD(
            [
                {'params': head_params, 'lr': init_lr_fit},
                {'params': other_params, 'lr': init_lr_fit * train_config.lr_rate}  
            ], 
            momentum=float(train_config.momentum), 
            nesterov=True,
            weight_decay=float(train_config.weight_decay))
                    }[train_config.optimizer_type]
        
    else:
        optimizer = {
        'adam': optim.Adam(trainable_params, init_lr_fit, betas=(float(train_config.momentum), 0.999),
                           weight_decay=float(train_config.weight_decay)),
        'sgd': optim.SGD(trainable_params, init_lr_fit, momentum=float(train_config.momentum), nesterov=True,
                         weight_decay=float(train_config.weight_decay))
        }[train_config.optimizer_type]

    #  lr_schedule
    lr_scheduler_func = get_lr_scheduler(train_config.lr_decay_type, init_lr_fit, min_lr_fit, train_config.epoch, 
                                         cycles=train_config.cycles, warmup_iters_ratio=train_config.warmup_iters_ratio, warmup_lr_ratio=train_config.warmup_lr_ratio,
                                            no_aug_iter_ratio=train_config.no_aug_iter_ratio, step_num=train_config.step_num,
                                         restart_lr_ratio=train_config.restart_lr_ratio)

    if global_config.sync_bn and ngpus_per_node > 1 and global_config.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif global_config.sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if global_config.distributed:

        model = model.cuda(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          find_unused_parameters=True)
    else:
        if not device == 'cpu':
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True
        model = model.to(device)

    if model_config.fp16:
        scaler = torch.amp.GradScaler(growth_interval=100)
    else:
        scaler = None

    # main process
    for epoch in range(int(train_config.init_Epoch), int(train_config.epoch)):

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)


        print("-"*40, epoch+1, "-"*40)

 
        model.train()
        if global_config.fine_tune:
            if not device == 'cpu':
                fix_frozen_bn(model.module.backbone)   # 注意 DDP 需要 .module
            else:
                fix_frozen_bn(model.backbone)


                
        if (epoch+1) % dataset_config.lfw_eval_period == 0:
            lfw_eval_period = True
        else:
             
            lfw_eval_period = False


        if global_config.cls_flag and not global_config.distill_flag and not global_config.semisiamese_distill_flag:  # only cls
            # print('only cls')
            fit_one_epoch(model, loss_history, optimizer, epoch, train_loader, val_loader, device, lfw_loader,
                        global_config.eval_flag,
                        model_config.fp16, scaler, train_config.save_period, loss_history.log_dir, lfw_eval_period, local_rank)
        
        elif not global_config.cls_flag and global_config.distill_flag and not global_config.semisiamese_distill_flag:  # only distill
            # print('only distill')
            distill(global_config, model, loss_history, optimizer, epoch, train_loader, distill_train_loader, semisiame_train_loader, val_loader, datapair_val_loader,
                  device, lfw_loader, face_data_loader, enface_data_loader, global_config.eval_flag, model_config.fp16, scaler, train_config.save_period, loss_history.log_dir, lfw_eval_period, triplet_loss_fn, model_teacher=model_teacher)
        
        elif global_config.cls_flag and global_config.distill_flag and not global_config.semisiamese_distill_flag:  # cls + distill
            # print('cls + distill')
            fit_with_distill(global_config, model, model_teacher, loss_history, optimizer, epoch, train_loader, val_loader, device, lfw_loader, distill_train_loader,
                        global_config.eval_flag,
                        model_config.fp16, scaler, train_config.save_period, loss_history.log_dir, lfw_eval_period, local_rank)
        
        elif global_config.semisiamese_distill_flag:  # cls + distill + semi
            # print('cls + distill + semi')
            fit_with_semi_siamese_v2(global_config, model, loss_history, optimizer, epoch, train_loader, distill_train_loader, semisiame_train_loader, val_loader, datapair_val_loader,
                  device, lfw_loader, face_data_loader, enface_data_loader, global_config.eval_flag, model_config.fp16, scaler, train_config.save_period, loss_history.log_dir, lfw_eval_period, triplet_loss_fn, model_teacher=model_teacher)

            
        if epoch < int(train_config.epoch) - 1:
            if dataset_config.dali:
                train_loader.reset()
                val_loader.reset()
        

    if loss_history.fpr is not None:
        plot_roc(loss_history.fpr, loss_history.tpr,
                 figure_name=os.path.join(loss_history.log_dir, global_config.roc_path))

    if local_rank == 0:
        loss_history.loss_plot(global_config.loss_path, global_config.acc_path)
        loss_history.writer.close()


# val 
def val(cfg_path, **kwargs):
    config = yaml.load(open(cfg_path, encoding='utf8'), Loader=yaml.FullLoader)
    config = EasyDict(config)  # convert to dict
    model_config = config.Arch

    global_config = config.Global
    dataset_config = config.Dataset
    seed_everything(global_config.seed)
    device = global_config.device

    
    model = Arcface(model_config.loss, model_config.margin_list, model_config.interclass_filtering_threshold,
                    model_config.embedding_size, 1000, model_config.sample_rate, model_config.fp16,
                    model_config.backbone, model_config.pretrained, global_config.distributed, device, **kwargs)




    if model_config.pretrained and not model_config.resume: 
        print('load from pretrained not resume')
        if model_config.backbone in [ 'mobilenetv1']: # mbf mobilenetv1
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_config.pretrained, map_location='cpu')
            # 重命名键：将开头 "arcface" 替换为 "backbone"
            new_pretrained_dict = {}
            for key, value in pretrained_dict.items():
                if key.startswith('arcface'):
                    new_key = key.replace('arcface', 'backbone', 1)  # 只替换第一次出现
                    new_pretrained_dict[new_key] = value
                else:
                    new_pretrained_dict[key] = value


            model_dict.update(new_pretrained_dict)
            model.load_state_dict(model_dict, strict=True)
        else: # r18 ... ...
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_config.pretrained, map_location='cpu')

            pretrained_dict = {f'backbone.{k}': v for k, v in pretrained_dict.items()}

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=True)

    elif model_config.pretrained and model_config.resume:
        print('load from pretrained with resume')
        print(model_config.pretrained)
        model_dict = torch.load(model_config.pretrained, map_location='cpu')
        backbone_dict = model_dict['state_dict_backbone']
        backbone_dict = {f'backbone.{k}': v for k, v in backbone_dict.items()}
        model.load_state_dict(backbone_dict, strict=False)


    model = model.to(device)

    if dataset_config.lfw_eval_flag:
        lfw_data = LFWDataset(dir=dataset_config.lfw_dir_path, pairs_path=dataset_config.lfw_pairs_path,
                              image_size=dataset_config.input_shape)
        data_loader = torch.utils.data.DataLoader(
            lfw_data,
            batch_size=dataset_config.lfw_batch_size,
            shuffle=dataset_config.lfw_shuffle,
            num_workers=dataset_config.num_workers,
            collate_fn=lfw_collate,
        )

    elif dataset_config.dataset_val_flag:
        semisiame_val_data = PairDataset(dir=dataset_config.dataset_dir_path, pairs_path=dataset_config.dataset_pairs_path,
                              image_size=dataset_config.input_shape)
        data_loader = DataLoader(
            semisiame_val_data,
            batch_size=dataset_config.dataset_batch_size,
            shuffle=dataset_config.dataset_shuffle,
            num_workers=dataset_config.num_workers,
            collate_fn=lfw_collate,
        )

    

    val_run(model, device, data_loader)


if __name__ == "__main__":

    train('config/config_train.yaml')
