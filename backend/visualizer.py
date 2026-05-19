"""
可视化模块
绘制训练/验证损失曲线、ROC曲线、混淆矩阵
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
)


class Visualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_losses(self, train_losses, val_losses, filename: str = "loss_curve.png") -> str:
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, "b-", label="训练损失", linewidth=2)
        plt.plot(epochs, val_losses, "r-", label="验证损失", linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("训练/验证损失曲线", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"损失曲线已保存至: {path}")
        return path

    def plot_roc(self, y_true, y_score, filename: str = "roc_curve.png", pos_label: int = 1) -> str:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        if y_score.ndim == 2:
            scores = y_score[:, pos_label]
        else:
            scores = y_score
        fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC 曲线 (AUC = {roc_auc:.4f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("Receiver Operating Characteristic (ROC)", fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"ROC曲线已保存至: {path}")
        return path

    def plot_confusion_matrix(
        self,
        y_true,
        y_pred,
        filename: str = "confusion_matrix.png",
        labels: list = None,
    ) -> str:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        cm = confusion_matrix(y_true, y_pred)

        if labels is None:
            unique = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
            labels = [str(x) for x in unique]

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=labels,
            yticklabels=labels,
            title="混淆矩阵",
            ylabel="真实标签",
            xlabel="预测标签",
        )

        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12,
                )

        fig.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"混淆矩阵已保存至: {path}")
        return path
