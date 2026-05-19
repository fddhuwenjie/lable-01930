"""
可视化模块
负责绘制 loss 曲线、ROC 曲线和混淆矩阵
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from typing import List, Optional


class Visualizer:
    def plot_loss_curve(self, train_losses: List[float], val_losses: List[float],
                        save_path: Optional[str] = None) -> None:
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)

        plt.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('训练/验证损失曲线', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"损失曲线已保存至: {save_path}")

        plt.close()

    def plot_roc_curve(self, y_true: List[int], y_scores: List[float],
                       save_path: Optional[str] = None) -> None:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC 曲线 (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='随机基线')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率 (FPR)', fontsize=12)
        plt.ylabel('真阳性率 (TPR)', fontsize=12)
        plt.title('ROC 曲线', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"ROC曲线已保存至: {save_path}")

        plt.close()

    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int],
                              labels: Optional[List[str]] = None,
                              save_path: Optional[str] = None) -> None:
        if labels is None:
            labels = ['正常', '恶意']

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im)
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, fontsize=12)
        plt.yticks(tick_marks, labels, fontsize=12)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha='center', va='center',
                         color='white' if cm[i, j] > thresh else 'black',
                         fontsize=14)

        plt.ylabel('真实标签', fontsize=12)
        plt.xlabel('预测标签', fontsize=12)
        plt.title('混淆矩阵', fontsize=14)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"混淆矩阵已保存至: {save_path}")

        plt.close()
