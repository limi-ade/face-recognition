import itertools
import os

import cv2
import numpy as np
from .dataloader import session
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from .utils import align_face_efficient, get_largest_face, get_lr, resize_image
from .utils_metrics import evaluate, plot_roc

def extrace_emb(img_BGR):

    faces = get_largest_face(session, img_BGR) # 找到最大人脸

    if faces is not None:
        dense_points = session.get_face_dense_landmark(faces) # 关键点
        aligend_dense = align_face_efficient(img_BGR, dense_points)  # 多点校准

        image = resize_image(aligend_dense, [112, 112], letterbox_image=True) # BGR
    else:
        image = resize_image(img_BGR, [112, 112], letterbox_image=True) # BGR
    
    try:
        face_detect = session.face_detection(image)

        face = face_detect[0]
        feature = session.face_feature_extract(image, face)
    
    except:
        face_detect = session.face_detection(img_BGR)

        face = face_detect[0]
        feature = session.face_feature_extract(image, face)

    return feature
                                            



def _ensure_nchw_batch(x):
    """LFWDataset 返回 (C,H,W)；batch_size=1 时 collate 可能仍为 3D，backbone 需要 (N,C,H,W)。"""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return x


def fit_one_epoch(model, loss_history, optimizer, epoch, gen, gen_val,
                  device, lfw_loader, lfw_eval_flag, fp16, scaler, save_period, save_dir, lfw_eval_period, local_rank=0):
    train_loss = 0
    train_accuracy = 0

    val_loss = 0
    val_accuracy = 0

    lfw_accuracy = 0
    lfw_val = 0

    lr = get_lr(optimizer)



    train_bar = tqdm(gen, colour='magenta', dynamic_ncols=True)
    train_bar.set_description('train')
    n = 0
    for i, batch in enumerate(train_bar):
        n += 1
        optimizer.zero_grad()
        images, labels = batch
        if not device == 'cpu':
            images = images.to(local_rank)
            labels = labels.to(local_rank)

        if fp16:
            with torch.amp.autocast():
                outputs, loss = model(images, labels, mode="train")
        else:
            outputs, loss = model(images, labels, mode="train")

        # 必须在 backward 之前统计：DistCrossEntropy 会原地修改 logits，
        # backward 后 outputs 已被破坏，准确率会恒为 0。
        with torch.no_grad():
            labels_flat = labels.view(-1)
            preds = torch.argmax(outputs, dim=-1)
            accuracy = (preds == labels_flat).float().mean()

        if fp16:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss += loss.item()
        train_accuracy += accuracy.item()

        if local_rank == 0:
            train_bar.set_postfix(**{'train_loss': train_loss / n,
                                     'train_accuracy': train_accuracy / n,
                                     'lr': get_lr(optimizer)})

    train_loss /= n
    train_accuracy /= n

    val_pbar = tqdm(gen_val, colour='green', dynamic_ncols=True)
    val_pbar.set_description('val')
    model.eval()
    with torch.no_grad():
        n = 0
        for batch in val_pbar:
            n += 1
            images, labels = batch
            if not device == 'cpu':
                images = images.to(local_rank)
                labels = labels.to(local_rank)

            outputs, loss = model(images, labels, mode="val")
            labels_flat = labels.view(-1)
            preds = torch.argmax(outputs, dim=-1)
            accuracy = (preds == labels_flat).float().mean()
            val_loss += loss.item()
            val_accuracy += accuracy.item()

            if local_rank == 0:
                val_pbar.set_postfix(**{'val_loss': val_loss / n,
                                        'val_accuracy': val_accuracy / n})

        val_loss /= n
        val_accuracy /= n

        if lfw_eval_flag and lfw_eval_period:
            test_pbar = tqdm(lfw_loader, colour='blue', dynamic_ncols=True)
            test_pbar.set_description('test')
            labels, distances = [], []
            for _, (data_a, data_p, label) in enumerate(test_pbar):
                if data_a is None:
                    continue
                data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
                data_a, data_p = _ensure_nchw_batch(data_a), _ensure_nchw_batch(data_p)
                data_a, data_p = data_a.to(device), data_p.to(device)
                out_a, out_p = model(data_a), model(data_p)
                dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
                distances.append(dists.data.cpu().numpy())
                labels.append(label.data.cpu().numpy())

            labels = np.array([sublabel for label in labels for sublabel in label])
            distances = np.array([subdist for dist in distances for subdist in dist])
            tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances, labels)
            print('lfw_Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
            print('lfw_Best_thresholds: %2.5f' % best_thresholds)
            print('lfw_Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
            lfw_accuracy += accuracy
            lfw_val += val
            loss_history.tpr = tpr
            loss_history.fpr = fpr

    if local_rank == 0:
        loss_history.append_loss(
                                epoch, lr, train_loss, 0, 0, 0, 
                                 0, 0,  train_accuracy, val_loss,  val_accuracy,  
                                 0, float(np.mean(lfw_accuracy)), lfw_val, 0, 0, 0, 0)
    if (epoch + 1) % save_period == 0:
        # 提取原始模型（如果被 DataParallel 包装）
        raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model

        backbone_sd = raw_model.backbone.state_dict()

        fc_sd = raw_model.module_partial_fc.state_dict()   # 根据实际属性名修改
        
        checkpoint = {
            "epoch": epoch + 1,
            "state_dict_backbone": backbone_sd,
            "state_dict_softmax_fc": fc_sd,
            "state_optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{epoch}.pt"))


def fit_with_distill(global_config, student, teacher, loss_history, optimizer, epoch, gen, gen_val,
                  device, lfw_loader, distill_loader, lfw_eval_flag, fp16, scaler, save_period, save_dir, lfw_eval_period, local_rank=0):
    train_loss = 0
    loss_cls_total = 0
    loss_mse_total = 0

    train_accuracy = 0

    val_loss = 0
    val_accuracy = 0

    lfw_accuracy = 0
    lfw_val = 0
    sim = 0

    length_train = len(gen)
    length_distill = len(distill_loader)

    lr = get_lr(optimizer)

    # 两个 loader 都必须可循环：步数取 max 时，较短的一方会先用尽；
    # 仅包装较短的一方且 DALI 未 reset 时，会在约 len(short)/len(long) 进度处阻塞。
    iter_gen = infinite_loader(gen)
    iter_distill = infinite_loader(distill_loader)
    total_steps = max(length_train, length_distill)

    train_bar = tqdm(range(total_steps), colour='magenta', dynamic_ncols=True)
    train_bar.set_description('train')
    n = 0
    for i, _ in enumerate(train_bar):
        n += 1
        optimizer.zero_grad()
        train_images, train_labels = next(iter_gen)
        distill_images, _ = next(iter_distill)

        if not device == 'cpu':
            train_images = train_images.to(local_rank)
            train_labels = train_labels.to(local_rank)

            distill_images = distill_images.to(local_rank)


        if fp16:
            with torch.amp.autocast():
                outputs, loss_cls = student(train_images, train_labels, mode="train")
                s_feat = student(distill_images)
        else:
            outputs, loss_cls = student(train_images, train_labels, mode="train")
            s_feat = student(distill_images)

        # 必须在 backward 之前统计：DistCrossEntropy 会原地修改 logits，
        with torch.no_grad():
            t_feat = teacher(distill_images)
            
            labels_flat = train_labels.view(-1)      # 微调模型准确度计算
            preds = torch.argmax(outputs, dim=-1)
            accuracy = (preds == labels_flat).float().mean()
        
        loss_distill = F.mse_loss(s_feat, t_feat)
        cos_sim = F.cosine_similarity(s_feat.cpu(), t_feat.cpu(), dim=1).mean().item()


        loss = global_config.cls_weight * loss_cls + global_config.distill_weight * loss_distill 

        if fp16:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss += loss.item()
        loss_cls_total += global_config.cls_weight * loss_cls.item()
        loss_mse_total += global_config.distill_weight * loss_distill.item()
        train_accuracy += accuracy.item()
        sim += cos_sim

        if local_rank == 0:
            train_bar.set_postfix(**{'loss': train_loss / n,
                                     'cls': loss_cls_total / n,
                                     'distill': loss_mse_total / n,
                                     'accuracy': train_accuracy / n,
                                     'similarity': sim / n,
                                     'lr': get_lr(optimizer)})

    train_loss /= n
    loss_cls_total /= n
    loss_mse_total /= n
    train_accuracy /= n
    sim /= n

    val_pbar = tqdm(gen_val, colour='green', dynamic_ncols=True)
    val_pbar.set_description('val')
    student.eval()
    with torch.no_grad():
        n = 0
        for batch in val_pbar:
            n += 1
            images, labels = batch
            if not device == 'cpu':
                images = images.to(local_rank)
                labels = labels.to(local_rank)

            outputs, loss = student(images, labels, mode="val")
            labels_flat = labels.view(-1)
            preds = torch.argmax(outputs, dim=-1)
            accuracy = (preds == labels_flat).float().mean()
            val_loss += loss.item()
            val_accuracy += accuracy.item()

            if local_rank == 0:
                val_pbar.set_postfix(**{'val_loss': val_loss / n,
                                        'val_accuracy': val_accuracy / n})

        val_loss /= n
        val_accuracy /= n

        if lfw_eval_flag and lfw_eval_period:
            test_pbar = tqdm(lfw_loader, colour='blue', dynamic_ncols=True)
            test_pbar.set_description('test')
            labels, distances = [], []
            for _, (data_a, data_p, label) in enumerate(test_pbar):
                if data_a is None:
                    continue
                data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
                data_a, data_p = _ensure_nchw_batch(data_a), _ensure_nchw_batch(data_p)
                data_a, data_p = data_a.to(device), data_p.to(device)
                out_a, out_p = student(data_a), student(data_p)
                dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
                distances.append(dists.data.cpu().numpy())
                labels.append(label.data.cpu().numpy())

            labels = np.array([sublabel for label in labels for sublabel in label])
            distances = np.array([subdist for dist in distances for subdist in dist])
            tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances, labels)
            print('lfw_Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
            print('lfw_Best_thresholds: %2.5f' % best_thresholds)
            print('lfw_Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
            lfw_accuracy += accuracy
            lfw_val += val
            loss_history.tpr = tpr
            loss_history.fpr = fpr

    if local_rank == 0:
        loss_history.append_loss(
                                epoch, lr, train_loss, loss_cls_total, 0, loss_mse_total, 
                                 0, sim,  train_accuracy, val_loss,  val_accuracy,  
                                 0, float(np.mean(lfw_accuracy)), lfw_val, 0, 0, 0, 0)
    if (epoch + 1) % save_period == 0:
        # 提取原始模型（如果被 DataParallel 包装）
        raw_model = student.module if isinstance(student, torch.nn.DataParallel) else student

        backbone_sd = raw_model.backbone.state_dict()

        fc_sd = raw_model.module_partial_fc.state_dict()   # 根据实际属性名修改
        
        checkpoint = {
            "epoch": epoch + 1,
            "state_dict_backbone": backbone_sd,
            "state_dict_softmax_fc": fc_sd,
            "state_optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{epoch}.pt"))

def fit_with_semi_siamese_v2(global_config, student, loss_history, optimizer, epoch, main_train_loader, distill_loader, semisiame_train_loader, main_val_loader, datapair_val_loader,
                  device, lfw_loader, face_loader, enface_loader, lfw_eval_flag, fp16, scaler, save_period, save_dir, lfw_eval_period, triplet_loss_fn, local_rank=0, model_teacher=False):

    train_loss = 0
    loss_cls_total = 0 
    loss_tripet_totall = 0
    loss_ss_total = 0
    loss_distill_total = 0
    sim = 0

    train_accuracy = 0

    val_dataset_loss = 0
    val_dataset_accuracy = 0
    val_datapair_accuracy = 0

    lfw_accuracy = 0
    lfw_val = 0
    
    face_accuracy = 0
    face_val = 0

    enface_accuracy = 0
    enface_val = 0

    length_train_cls = len(main_train_loader)
    length_train_ss = len(semisiame_train_loader)
    length_train_distill = len(distill_loader)

    lr = get_lr(optimizer)

    iter_main = infinite_loader(main_train_loader)
    iter_semisiame = infinite_loader(semisiame_train_loader)
    iter_distill = infinite_loader(distill_loader)
    total_steps = max(length_train_cls, length_train_ss, length_train_distill)

    train_bar = tqdm(range(total_steps), colour='magenta', dynamic_ncols=True)
    train_bar.set_description('train')
    n = 0
    for i, _ in enumerate(train_bar):
        n += 1
        optimizer.zero_grad()
        train_main_images, train_main_labels = next(iter_main)
        view1, view2 = next(iter_semisiame)  # 两个增强视图
        distill_images, image_BGR = next(iter_distill)

        if not device == 'cpu':
            train_main_images = train_main_images.to(local_rank)
            train_main_labels = train_main_labels.to(local_rank)
            view1 = view1.to(local_rank) # 强
            view2 = view2.to(local_rank) # 弱
            distill_images = distill_images.to(local_rank)

        if fp16:
            with torch.amp.autocast():
                outputs, loss_cls = student(train_main_images, train_main_labels, mode="train")    # 分类损失                    
                
                emb_train = student(train_main_images)                  # 获取原始 embedding
                loss_triplet = triplet_loss_fn(emb_train, train_main_labels)       # 三元组损失（仅使用原始图像，并且需要 L2 归一化）
                
                s_feat = student(distill_images)  # 学生 蒸馏 信号

                ss_weak = student(view2)  # 学生 弱孪生 信号

        else:
            outputs, loss_cls = student(train_main_images, train_main_labels, mode="train")    # 分类损失                     
                
            emb_train = student(train_main_images)                  # 获取原始 embedding
            loss_triplet = triplet_loss_fn(emb_train, train_main_labels)       # 三元组损失（仅使用原始图像，并且需要 L2 归一化）
            
            s_feat = student(distill_images)  # 学生 蒸馏 信号
            
            ss_weak = student(view2)  # 学生 弱孪生 信号

        with torch.no_grad():
            if model_teacher is not False:
                t_feat = model_teacher(distill_images)
            else:

                features = []
                for img in image_BGR.numpy():
                    feature = extrace_emb(img)
                    features.append(feature)
                
                t_feat = np.stack(features, axis=0) # 基线 蒸馏 信号

            t_feat = t_feat.to(device) if isinstance(t_feat, torch.Tensor) else torch.from_numpy(t_feat)
            # t_feat = F.normalize(t_feat, p=2, dim=1)
            ss_strong = student(view1)  # 学生 强孪生 信号
      
        loss_distill = F.mse_loss(s_feat,t_feat) 
        loss_semi = F.mse_loss(ss_weak, ss_strong)


        labels_flat = train_main_labels.view(-1)      
        preds = torch.argmax(outputs, dim=-1)
        accuracy = (preds == labels_flat).float().mean()  # 微调模型准确度计算
        cos_sim = F.cosine_similarity(s_feat.cpu(),t_feat.cpu(), dim=1).mean().item() # 蒸馏相似度计算
        train_accuracy += accuracy.item()

        # 总损失
        loss =  global_config.cls_weight* loss_cls + global_config.distill_weight * loss_distill + global_config.triplet_weight * loss_triplet + global_config.semi_weight * loss_semi
     


        if fp16:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5)
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

        # 统计
        train_loss += loss.item()
        loss_ss_total += global_config.semi_weight * loss_semi.item()
        loss_distill_total += global_config.distill_weight * loss_distill.item()
        loss_cls_total += loss_cls.item() 
        loss_tripet_totall += global_config.triplet_weight * loss_triplet.item()
        sim += cos_sim

        if local_rank == 0:
            train_bar.set_postfix(**{'loss': train_loss / n,
                                     'ss': loss_ss_total / n,
                                     'dis': loss_distill_total / n,
                                     'cls': loss_cls_total / n,
                                     'tri': loss_tripet_totall / n,
                                     'sim': sim / n,
                                     'lr': get_lr(optimizer)})

    train_loss /= n
    loss_ss_total /= n
    loss_distill_total /= n
    loss_cls_total /= n  
    loss_tripet_totall /= n
    train_accuracy /= n
    sim /= n

    student.eval()
    with torch.no_grad():
        # val_dataset
        val_pbar = tqdm(main_val_loader, colour='green', dynamic_ncols=True)
        val_pbar.set_description('val_dataset')   
        n = 0
        for batch in val_pbar:
            n += 1
            images, labels = batch
            if not device == 'cpu':
                images = images.to(local_rank)
                labels = labels.to(local_rank)

            outputs, loss = student(images, labels, mode="val")
            labels_flat = labels.view(-1)
            preds = torch.argmax(outputs, dim=-1)
            accuracy = (preds == labels_flat).float().mean()
            val_dataset_loss += loss.item()
            val_dataset_accuracy += accuracy.item()

            if local_rank == 0:
                val_pbar.set_postfix(**{'val_loss': val_dataset_loss / n,
                                        'val_accuracy': val_dataset_accuracy / n})

        val_dataset_loss /= n
        val_dataset_accuracy /= n

        # datapair
        val_pbar = tqdm(datapair_val_loader, colour='yellow', dynamic_ncols=True)
        val_pbar.set_description('val_dadapair')
        labels, distances = [], []
        for _, (data_a, data_p, label) in enumerate(val_pbar):
            if data_a is None:
                    continue
            data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
            data_a, data_p = _ensure_nchw_batch(data_a), _ensure_nchw_batch(data_p)
            data_a, data_p = data_a.to(device), data_p.to(device)
            out_a, out_p = student(data_a), student(data_p)
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
            distances.append(dists.data.cpu().numpy())
            labels.append(label.data.cpu().numpy())

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])
        tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances, labels)
        print('val_Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
        print('val_Best_thresholds: %2.5f' % best_thresholds)
        print('val_Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
        val_datapair_accuracy += float(np.mean(accuracy))

        if lfw_eval_flag and lfw_eval_period:
            # LFW
            test_pbar = tqdm(lfw_loader, colour='blue', dynamic_ncols=True)
            test_pbar.set_description('lfw')
            labels, distances = [], []
            for _, (data_a, data_p, label) in enumerate(test_pbar):
                if data_a is None:
                    continue
                data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
                data_a, data_p = _ensure_nchw_batch(data_a), _ensure_nchw_batch(data_p)
                data_a, data_p = data_a.to(device), data_p.to(device)
                out_a, out_p = student(data_a), student(data_p)
                dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
                distances.append(dists.data.cpu().numpy())
                labels.append(label.data.cpu().numpy())

            labels = np.array([sublabel for label in labels for sublabel in label])
            distances = np.array([subdist for dist in distances for subdist in dist])
            tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances, labels)
            print('lfw_Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
            print('lfw_Best_thresholds: %2.5f' % best_thresholds)
            print('lfw_Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
            lfw_accuracy += np.mean(accuracy)
            lfw_val += val

            # face
            test_pbar = tqdm(face_loader, colour='cyan', dynamic_ncols=True)
            test_pbar.set_description('face')
            labels, distances = [], []
            for _, (data_a, data_p, label) in enumerate(test_pbar):
                if data_a is None:
                    continue
                data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
                data_a, data_p = _ensure_nchw_batch(data_a), _ensure_nchw_batch(data_p)
                data_a, data_p = data_a.to(device), data_p.to(device)
                out_a, out_p = student(data_a), student(data_p)
                dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
                distances.append(dists.data.cpu().numpy())
                labels.append(label.data.cpu().numpy())

            labels = np.array([sublabel for label in labels for sublabel in label])
            distances = np.array([subdist for dist in distances for subdist in dist])
            tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances, labels)
            print('face_Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
            print('face_Best_thresholds: %2.5f' % best_thresholds)
            print('face_Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
            face_accuracy += np.mean(accuracy)
            face_val += val

            loss_history.tpr = tpr
            loss_history.fpr = fpr

            # enface
            test_pbar = tqdm(enface_loader, colour='red', dynamic_ncols=True)
            test_pbar.set_description('enface')
            labels, distances = [], []
            for _, (data_a, data_p, label) in enumerate(test_pbar):
                if data_a is None:
                    continue
                data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
                data_a, data_p = _ensure_nchw_batch(data_a), _ensure_nchw_batch(data_p)
                data_a, data_p = data_a.to(device), data_p.to(device)
                out_a, out_p = student(data_a), student(data_p)
                dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
                distances.append(dists.data.cpu().numpy())
                labels.append(label.data.cpu().numpy())

            labels = np.array([sublabel for label in labels for sublabel in label])
            distances = np.array([subdist for dist in distances for subdist in dist])
            tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances, labels)
            print('lfw_Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
            print('lfw_Best_thresholds: %2.5f' % best_thresholds)
            print('lfw_Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
            enface_accuracy += np.mean(accuracy)
            enface_val += val

       

    if local_rank == 0:
        loss_history.append_loss(epoch, lr, train_loss, loss_cls_total, loss_tripet_totall, loss_distill_total, 
                                 loss_ss_total, sim,  train_accuracy, val_dataset_loss,  val_dataset_accuracy,  
                                 val_datapair_accuracy, lfw_accuracy, lfw_val, face_accuracy, face_val, enface_accuracy, enface_val)
    if (epoch + 1) % save_period == 0:
        # 提取原始模型（如果被 DataParallel 包装）
        raw_model = student.module if isinstance(student, torch.nn.DataParallel) else student

        backbone_sd = raw_model.backbone.state_dict()

        
        checkpoint = {
            "epoch": epoch + 1,
            "state_dict_backbone": backbone_sd,
            "state_optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{epoch}.pt"))



def distill(global_config, student, loss_history, optimizer, epoch, main_train_loader, distill_loader, semisiame_train_loader, main_val_loader, datapair_val_loader,
                  device, lfw_loader, face_loader, enface_loader, lfw_eval_flag, fp16, scaler, save_period, save_dir, lfw_eval_period, triplet_loss_fn, local_rank=0, model_teacher=False):

    train_loss = 0
    loss_cls_total = 0 
    loss_tripet_totall = 0
    loss_ss_total = 0
    loss_distill_total = 0
    sim = 0

    train_accuracy = 0

    val_dataset_loss = 0
    val_dataset_accuracy = 0
    val_datapair_accuracy = 0

    lfw_accuracy = 0
    lfw_val = 0
    
    face_accuracy = 0
    face_val = 0

    enface_accuracy = 0
    enface_val = 0

    # length_train_cls = len(main_train_loader)
    # length_train_ss = len(semisiame_train_loader)
    length_train_distill = len(distill_loader)

    lr = get_lr(optimizer)

    # iter_main = infinite_loader(main_train_loader)
    # iter_semisiame = infinite_loader(semisiame_train_loader)
    iter_distill = infinite_loader(distill_loader)
    total_steps = length_train_distill

    train_bar = tqdm(range(total_steps), colour='magenta', dynamic_ncols=True)
    train_bar.set_description('train')
    n = 0
    for i, _ in enumerate(train_bar):
        n += 1
        optimizer.zero_grad()
        # train_main_images, train_main_labels = next(iter_main)
        # view1, view2 = next(iter_semisiame)  # 两个增强视图
        distill_images, image_BGR = next(iter_distill)

        if not device == 'cpu':
            # train_main_images = train_main_images.to(local_rank)
            # train_main_labels = train_main_labels.to(local_rank)
            # view1 = view1.to(local_rank) # 强
            # view2 = view2.to(local_rank) # 弱
            distill_images = distill_images.to(local_rank)

        if fp16:
            with torch.amp.autocast():
                # outputs, loss_cls = student(train_main_images, train_main_labels, mode="train")    # 分类损失                    
                
                # emb_train = student(train_main_images)                  # 获取原始 embedding
                # loss_triplet = triplet_loss_fn(emb_train, train_main_labels)       # 三元组损失（仅使用原始图像，并且需要 L2 归一化）
                
                s_feat = student(distill_images)  # 学生 蒸馏 信号

                # ss_weak = student(view2)  # 学生 弱孪生 信号

        else:
            # outputs, loss_cls = student(train_main_images, train_main_labels, mode="train")    # 分类损失                     
                
            # emb_train = student(train_main_images)                  # 获取原始 embedding
            # loss_triplet = triplet_loss_fn(emb_train, train_main_labels)       # 三元组损失（仅使用原始图像，并且需要 L2 归一化）
            
            s_feat = student(distill_images)  # 学生 蒸馏 信号
            
            # ss_weak = student(view2)  # 学生 弱孪生 信号

        with torch.no_grad():
            if model_teacher is not False:
                t_feat = model_teacher(distill_images)
            else:

                features = []
                for img in image_BGR.numpy():
                    feature = extrace_emb(img)
                    features.append(feature)
                
                t_feat = np.stack(features, axis=0) # 基线 蒸馏 信号

            t_feat = t_feat.to(device) if isinstance(t_feat, torch.Tensor) else torch.from_numpy(t_feat)
            # t_feat = F.normalize(t_feat, p=2, dim=1)
            # ss_strong = student(view1)  # 学生 强孪生 信号
      
        loss_distill = F.mse_loss(s_feat,t_feat) 
        # loss_semi = F.mse_loss(ss_weak, ss_strong)


        # labels_flat = train_main_labels.view(-1)      
        # preds = torch.argmax(outputs, dim=-1)
        # accuracy = (preds == labels_flat).float().mean()  # 微调模型准确度计算
        cos_sim = F.cosine_similarity(s_feat.cpu(),t_feat.cpu(), dim=1).mean().item() # 蒸馏相似度计算
        # train_accuracy += accuracy.item()

        # 总损失
        loss = 1 * loss_distill
     


        if fp16:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5)
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

        # 统计
        train_loss += loss.item()
        # loss_ss_total += global_config.semi_weight * loss_semi.item()
        loss_distill_total += loss_distill.item()
        # loss_cls_total += loss_cls.item() 
        # loss_tripet_totall += global_config.triplet_weight * loss_triplet.item()
        sim += cos_sim

        if local_rank == 0:
            train_bar.set_postfix(**{'loss': train_loss / n,
                                     'ss': loss_ss_total / n,
                                     'dis': loss_distill_total / n,
                                     'cls': loss_cls_total / n,
                                     'tri': loss_tripet_totall / n,
                                     'sim': sim / n,
                                     'lr': get_lr(optimizer)})

    train_loss /= n
    loss_ss_total /= n
    loss_distill_total /= n
    loss_cls_total /= n  
    loss_tripet_totall /= n
    train_accuracy /= n
    sim /= n

    student.eval()
    with torch.no_grad():
        # # val_dataset
        # val_pbar = tqdm(main_val_loader, colour='green', dynamic_ncols=True)
        # val_pbar.set_description('val_dataset')   
        # n = 0
        # for batch in val_pbar:
        #     n += 1
        #     images, labels = batch
        #     if not device == 'cpu':
        #         images = images.to(local_rank)
        #         labels = labels.to(local_rank)

        #     outputs, loss = student(images, labels, mode="val")
        #     labels_flat = labels.view(-1)
        #     preds = torch.argmax(outputs, dim=-1)
        #     accuracy = (preds == labels_flat).float().mean()
        #     val_dataset_loss += loss.item()
        #     val_dataset_accuracy += accuracy.item()

        #     if local_rank == 0:
        #         val_pbar.set_postfix(**{'val_loss': val_dataset_loss / n,
        #                                 'val_accuracy': val_dataset_accuracy / n})

        # val_dataset_loss /= n
        # val_dataset_accuracy /= n

        # datapair
        val_pbar = tqdm(datapair_val_loader, colour='yellow', dynamic_ncols=True)
        val_pbar.set_description('val_dadapair')
        labels, distances = [], []
        for _, (data_a, data_p, label) in enumerate(val_pbar):
            if data_a is None:
                    continue
            data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
            data_a, data_p = _ensure_nchw_batch(data_a), _ensure_nchw_batch(data_p)
            data_a, data_p = data_a.to(device), data_p.to(device)
            out_a, out_p = student(data_a), student(data_p)
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
            distances.append(dists.data.cpu().numpy())
            labels.append(label.data.cpu().numpy())

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])
        tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances, labels)
        print('val_Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
        print('val_Best_thresholds: %2.5f' % best_thresholds)
        print('val_Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
        val_datapair_accuracy += float(np.mean(accuracy))

        if lfw_eval_flag and lfw_eval_period:
            # LFW
            test_pbar = tqdm(lfw_loader, colour='blue', dynamic_ncols=True)
            test_pbar.set_description('lfw')
            labels, distances = [], []
            for _, (data_a, data_p, label) in enumerate(test_pbar):
                if data_a is None:
                    continue
                data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
                data_a, data_p = _ensure_nchw_batch(data_a), _ensure_nchw_batch(data_p)
                data_a, data_p = data_a.to(device), data_p.to(device)
                out_a, out_p = student(data_a), student(data_p)
                dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
                distances.append(dists.data.cpu().numpy())
                labels.append(label.data.cpu().numpy())

            labels = np.array([sublabel for label in labels for sublabel in label])
            distances = np.array([subdist for dist in distances for subdist in dist])
            tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances, labels)
            print('lfw_Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
            print('lfw_Best_thresholds: %2.5f' % best_thresholds)
            print('lfw_Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
            lfw_accuracy += np.mean(accuracy)
            lfw_val += val

            # face
            test_pbar = tqdm(face_loader, colour='cyan', dynamic_ncols=True)
            test_pbar.set_description('face')
            labels, distances = [], []
            for _, (data_a, data_p, label) in enumerate(test_pbar):
                if data_a is None:
                    continue
                data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
                data_a, data_p = _ensure_nchw_batch(data_a), _ensure_nchw_batch(data_p)
                data_a, data_p = data_a.to(device), data_p.to(device)
                out_a, out_p = student(data_a), student(data_p)
                dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
                distances.append(dists.data.cpu().numpy())
                labels.append(label.data.cpu().numpy())

            labels = np.array([sublabel for label in labels for sublabel in label])
            distances = np.array([subdist for dist in distances for subdist in dist])
            tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances, labels)
            print('face_Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
            print('face_Best_thresholds: %2.5f' % best_thresholds)
            print('face_Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
            face_accuracy += np.mean(accuracy)
            face_val += val

            loss_history.tpr = tpr
            loss_history.fpr = fpr

            # enface
            test_pbar = tqdm(enface_loader, colour='red', dynamic_ncols=True)
            test_pbar.set_description('enface')
            labels, distances = [], []
            for _, (data_a, data_p, label) in enumerate(test_pbar):
                if data_a is None:
                    continue
                data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
                data_a, data_p = _ensure_nchw_batch(data_a), _ensure_nchw_batch(data_p)
                data_a, data_p = data_a.to(device), data_p.to(device)
                out_a, out_p = student(data_a), student(data_p)
                dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
                distances.append(dists.data.cpu().numpy())
                labels.append(label.data.cpu().numpy())

            labels = np.array([sublabel for label in labels for sublabel in label])
            distances = np.array([subdist for dist in distances for subdist in dist])
            tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances, labels)
            print('lfw_Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
            print('lfw_Best_thresholds: %2.5f' % best_thresholds)
            print('lfw_Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
            enface_accuracy += np.mean(accuracy)
            enface_val += val

       

    if local_rank == 0:
        loss_history.append_loss(epoch, lr, train_loss, loss_cls_total, loss_tripet_totall, loss_distill_total, 
                                 loss_ss_total, sim,  train_accuracy, val_dataset_loss,  val_dataset_accuracy,  
                                 val_datapair_accuracy, lfw_accuracy, lfw_val, face_accuracy, face_val, enface_accuracy, enface_val)
    if (epoch + 1) % save_period == 0:
        # 提取原始模型（如果被 DataParallel 包装）
        raw_model = student.module if isinstance(student, torch.nn.DataParallel) else student

        backbone_sd = raw_model.backbone.state_dict()

        
        checkpoint = {
            "epoch": epoch + 1,
            "state_dict_backbone": backbone_sd,
            "state_optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{epoch}.pt"))





def infinite_loader(loader):
    """循环产出 batch；DALI 迭代器在 epoch 结束需 reset，否则下一轮会阻塞。"""
    while True:
        for batch in loader:
            yield batch
        if hasattr(loader, 'reset'):
            loader.reset()


def val_run(model, device, data_loader):
 
    labels, distances = [], []

    model.eval()
    with torch.no_grad():
        test_pbar = tqdm(data_loader, colour='blue', dynamic_ncols=True)
        test_pbar.set_description('test')
        
        for _, (data_a, data_p, label) in enumerate(test_pbar):
            if data_a is None:
                continue
            data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
            data_a, data_p = _ensure_nchw_batch(data_a), _ensure_nchw_batch(data_p)
            data_a, data_p = data_a.to(device), data_p.to(device)
            out_a, out_p = model(data_a), model(data_p)
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
            distances.append(dists.data.cpu().numpy())
            labels.append(label.data.cpu().numpy())

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])
        tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances, labels)
        print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
        print('Best_thresholds: %2.5f' % best_thresholds)
        print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
        plot_roc(fpr, tpr)