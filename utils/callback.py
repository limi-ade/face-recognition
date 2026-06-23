import datetime
import os
import shutil
import torch
import matplotlib

matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class LossHistory:
    def __init__(self, log_dir, cfg_path, model, input_shape):
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        self.log_dir = os.path.join(log_dir, cfg_path+"_" + str(time_str))

        self.lr = []
        self.train_losses = []
        self.loss_cls = []
        self.loss_tripet = []
        self.loss_distill = []
        self.loss_ss = []
        self.cos_sim = []
        self.train_acc = []


        self.val_dataset_loss = []   
        self.val_dataset_acc = []
        self.val_datapair_acc = []
        self.lfw_acc = []
        self.lfw_val = []
        self.face_acc = []
        self.face_val = []
        self.enface_acc = []
        self.enface_val = []


        os.makedirs(self.log_dir)
        shutil.copy(cfg_path, self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1]).to('cpu')
        self.writer.add_graph(model.backbone.to('cpu'), dummy_input)
        self.writer.flush()


        self.tpr, self.fpr = None, None

    def append_loss(self, epoch, lr, train_loss, loss_cls_total, loss_tripet_total, loss_distill_total, 
                                 loss_ss_total, sim,  train_accuracy, val_dataset_loss,  val_dataset_accuracy,  
                                 val_datapair_accuracy, lfw_accuracy, lfw_val, face_accuracy, face_val, enface_accuracy, enface_val):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.lr.append(lr)
        self.train_losses.append(train_loss)
        self.loss_cls.append(loss_cls_total)
        self.loss_tripet.append(loss_tripet_total)
        self.loss_distill.append(loss_distill_total)
        self.loss_ss.append(loss_ss_total)
        self.cos_sim.append(sim)
        self.train_acc.append(train_accuracy)

        self.val_dataset_loss.append(val_dataset_loss)
        self.val_dataset_acc.append(val_dataset_accuracy)  
        self.val_datapair_acc.append(val_datapair_accuracy)
        self.lfw_acc.append(lfw_accuracy)
        self.lfw_val.append(lfw_val)
        self.face_acc.append(face_accuracy)
        self.face_val.append(face_val)
        self.enface_acc.append(enface_accuracy)
        self.enface_val.append(enface_val)

        
        self.writer.add_scalar('lr', lr, epoch)
        self.writer.add_scalar('train_loss', train_loss, epoch)
        self.writer.add_scalar('loss_cls', loss_cls_total, epoch)
        self.writer.add_scalar('loss_tripet', loss_tripet_total, epoch)
        self.writer.add_scalar('loss_distill', loss_distill_total, epoch)
        self.writer.add_scalar('loss_ss', loss_ss_total, epoch)
        self.writer.add_scalar('cos_sim', sim, epoch)
        self.writer.add_scalar('train_acc', train_accuracy, epoch)

        self.writer.add_scalar('val_dataset_loss', val_dataset_loss, epoch)
        self.writer.add_scalar('val_dataset_acc', val_dataset_accuracy, epoch)
        self.writer.add_scalar('val_datapair_acc', val_datapair_accuracy, epoch)

        self.writer.add_scalar('lfw_acc', lfw_accuracy, epoch)
        self.writer.add_scalar('lfw_val', lfw_val, epoch)

        self.writer.add_scalar('face_acc', face_accuracy, epoch)
        self.writer.add_scalar('face_val', face_val, epoch)

        self.writer.add_scalar('enface_acc', enface_accuracy, epoch)
        self.writer.add_scalar('enface_val', enface_val, epoch)

    def loss_plot(self, loss_path, acc_path):
        # ------------------- 损失曲线（双轴）-------------------
        epochs_loss = range(len(self.train_losses))
        fig, ax1 = plt.subplots()
        ax1.plot(epochs_loss, self.train_losses, 'magenta', linewidth=2, label='train loss')
        ax1.plot(epochs_loss, self.val_dataset_loss, 'blue', linewidth=2, label='val dataset loss')
        ax1.plot(epochs_loss, self.loss_cls, 'cyan', linewidth=2, label='cls loss')
        ax1.plot(epochs_loss, self.loss_distill, 'green', linewidth=2, label='distill loss')
        ax1.plot(epochs_loss, self.loss_tripet, 'red', linewidth=2, label='tripet loss')
        ax1.plot(epochs_loss, self.loss_ss, 'black', linewidth=2, label='ss loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(epochs_loss, self.cos_sim, 'yellow', linewidth=2, label='cos_sim')
        ax2.set_ylabel('Cosine Similarity')

        # 合并两个轴的图例
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        fig.legend(handles, labels, loc='center', framealpha=0.9, edgecolor='black')

        plt.title('Training Curves')
        plt.savefig(os.path.join(self.log_dir, loss_path))
        plt.cla()
        plt.close("all")

        # ------------------- 准确率 + 学习率合并图（学习率左轴，准确率右轴）-------------------
        epochs_acc = range(len(self.train_acc))
        fig_acc, ax_lr = plt.subplots()      # 左轴：学习率
        ax_lr.plot(epochs_acc, self.lr, 'green', linewidth=2, label='learning rate')
        ax_lr.set_xlabel('Epoch')
        ax_lr.set_ylabel('Learning Rate')
        ax_lr.grid(True)

        ax_acc = ax_lr.twinx()               # 右轴：准确率
        ax_acc.plot(epochs_acc, self.train_acc, 'blue', linewidth=2, label='train acc')
        ax_acc.plot(epochs_acc, self.val_dataset_acc, 'orange', linewidth=2, label='val dataset acc')
        ax_acc.plot(epochs_acc, self.val_datapair_acc, 'magenta', linewidth=2, label='val datapair acc')
        ax_acc.set_ylabel('Accuracy')

        # 合并图例，放在右侧中间位置
        handles1, labels1 = ax_lr.get_legend_handles_labels()
        handles2, labels2 = ax_acc.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        fig_acc.legend(handles, labels, loc='center', framealpha=0.9, edgecolor='black')

        # 每10个epoch标注准确率数值
        for i, (acc_train, acc_dataset_val, acc_datapair_val) in enumerate(zip(self.train_acc, self.val_dataset_acc, self.val_datapair_acc)):
            if (i + 1) % 10 == 0:
                ax_acc.text(i, acc_train, f'{acc_train:.3f}', fontsize=8, ha='center', va='bottom')
                ax_acc.text(i, acc_dataset_val, f'{acc_dataset_val:.3f}', fontsize=8, ha='center', va='top')
                ax_acc.text(i, acc_datapair_val, f'{acc_datapair_val:.3f}', fontsize=8, ha='center', va='top')

        # 显示最终LFW准确率（文本框放在右轴坐标系）
        final_lfw_acc = self.lfw_acc[-1] if self.lfw_acc else 0.0
        ax_acc.text(0.02, 0.98, f'Final LFW Acc: {final_lfw_acc:.3f}',
                    transform=ax_acc.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'),
                    zorder=10)

        plt.title('Accuracy & Learning Rate')
        plt.savefig(os.path.join(self.log_dir, acc_path))
        plt.cla()
        plt.close("all")

        # ------------------- 新增 LFW 准确率曲线 -------------------
        if hasattr(self, 'lfw_acc') and self.lfw_acc and hasattr(self, 'lfw_val') and self.lfw_val:
            # 过滤掉 acc 或 val 为 0 的数据点，保持长度一致
            filtered = [(acc, val, face_acc, face_val,enface_acc, enface_val) for acc, val, face_acc, face_val,enface_acc, enface_val in zip(self.lfw_acc, self.lfw_val, self.face_acc, self.face_val, self.enface_acc, self.enface_val) if acc != 0 and val != 0]
        if filtered:
            lfw_acc_filtered, lfw_val_filtered, face_acc_filtered, face_val_filtered, enface_acc_filtered, enface_val_filtered = zip(*filtered)
            valid_indices = range(len(lfw_acc_filtered))
            
            plt.figure()
            plt.plot(valid_indices, lfw_acc_filtered, 'cyan', linewidth=2, marker='o', label='LFW Acc')
            plt.plot(valid_indices, lfw_val_filtered, 'orange', linewidth=2, marker='s', label='LFW Val')
            plt.plot(valid_indices, face_acc_filtered, 'blue', linewidth=2, marker='o', label='face Acc')
            plt.plot(valid_indices, face_val_filtered, 'green', linewidth=2, marker='s', label='face Val')
            plt.plot(valid_indices, enface_acc_filtered, 'magenta', linewidth=2, marker='o', label='enface Acc')
            plt.plot(valid_indices, enface_val_filtered, 'yellow', linewidth=2, marker='s', label='enface Val')


                        # 每五个点标注一次
            for i, (acc, val, face_acc, face_val, enface_acc, enface_val) in enumerate(zip(lfw_acc_filtered, lfw_val_filtered, face_acc_filtered, face_val_filtered, enface_acc_filtered, enface_val_filtered)):
                if (i+1) % 5 == 0:
                    # 为每一条线的 X 轴加上微小的偏移量（-0.15 到 +0.15），使文字水平错开
                    plt.text(i - 0.12, acc, f'{acc:.4f}', fontsize=7, rotation=45, ha='center', va='bottom')
                    plt.text(i - 0.08, val, f'{val:.4f}', fontsize=7, rotation=45, ha='center', va='bottom')
                    plt.text(i - 0.04, face_acc, f'{face_acc:.4f}', fontsize=7, rotation=45, ha='center', va='bottom')
                    plt.text(i + 0.04, face_val, f'{face_val:.4f}', fontsize=7, rotation=45, ha='center', va='bottom')
                    plt.text(i + 0.08, enface_acc, f'{enface_acc:.4f}', fontsize=7, rotation=45, ha='center', va='bottom')
                    plt.text(i + 0.12, enface_val, f'{enface_val:.4f}', fontsize=7, rotation=45, ha='center', va='bottom')
            
            plt.grid(True)
            plt.xlabel('Valid Checkpoint Index')
            plt.ylabel('Accuracy')
            plt.legend(loc='lower right')
            plt.title('LFW Accuracy (Non-zero Values)')
            plt.savefig(os.path.join(self.log_dir, 'lfw_curve.png'))
            plt.cla()
            plt.close('all')