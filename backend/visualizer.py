"""
可视化模块
负责绘制训练损失曲线、ROC 曲线和混淆矩阵
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from typing import List, Tuple, Optional
import os

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """
    可视化工具类
    """
    
    @staticmethod
    def plot_losses(
        train_losses: List[float],
        val_losses: List[float],
        save_path: str = None
    ) -> None:
        """
        绘制训练/验证损失曲线
        
        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            save_path: 图片保存路径
        """
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
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"损失曲线已保存至: {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: str = None
    ) -> float:
        """
        绘制 ROC 曲线
        
        Args:
            y_true: 真实标签
            y_proba: 预测概率（正类概率）
            save_path: 图片保存路径
            
        Returns:
            AUC 值
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 曲线 (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (False Positive Rate)', fontsize=12)
        plt.ylabel('真正率 (True Positive Rate)', fontsize=12)
        plt.title('ROC 曲线', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"ROC 曲线已保存至: {save_path}")
        
        plt.close()
        return roc_auc
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = None,
        save_path: str = None
    ) -> np.ndarray:
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称
            save_path: 图片保存路径
            
        Returns:
            混淆矩阵
        """
        if class_names is None:
            class_names = ['正常', '恶意']
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混淆矩阵', fontsize=14)
        plt.colorbar()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, fontsize=10)
        plt.yticks(tick_marks, class_names, fontsize=10)
        
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14)
        
        plt.ylabel('真实标签', fontsize=12)
        plt.xlabel('预测标签', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"混淆矩阵已保存至: {save_path}")
        
        plt.close()
        return cm
