"""
恶意URL分类模型训练器
封装训练、验证、early stopping、checkpoint保存等逻辑
"""
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Optional

from config import TrainConfig
from data_loader import URLDataLoader
from model import URLClassifier
from visualizer import Visualizer


class Trainer:
    """
    模型训练器，支持early stopping和checkpoint管理
    """

    def __init__(self, config: TrainConfig, model: nn.Module = None):
        self.config = config
        self.device = config.device
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []

        self._early_stop_counter = 0
        self._best_val_loss = float("inf")
        self._checkpoint_dir = os.path.join(config.output_dir, "checkpoints")

    def build_model(self, input_dim: int) -> nn.Module:
        self.model = URLClassifier(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            num_classes=self.config.num_classes,
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        return self.model

    def train_epoch(self, train_loader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches

    def validate(self, val_loader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def _should_early_stop(self, val_loss: float) -> bool:
        if self._best_val_loss - val_loss > self.config.early_stopping_min_delta:
            self._best_val_loss = val_loss
            self._early_stop_counter = 0
        else:
            self._early_stop_counter += 1
        return self._early_stop_counter >= self.config.early_stopping_patience

    def _save_checkpoint(self, epoch: int, val_loss: float, val_acc: float) -> None:
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(
            self._checkpoint_dir,
            f"checkpoint_epoch{epoch:03d}_loss{val_loss:.4f}.pt",
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            },
            ckpt_path,
        )

        existing = sorted(
            glob.glob(os.path.join(self._checkpoint_dir, "checkpoint_*.pt"))
        )
        if len(existing) > self.config.max_checkpoints:
            for old in existing[: -self.config.max_checkpoints]:
                os.remove(old)

    def _find_best_checkpoint(self):
        if not os.path.isdir(self._checkpoint_dir):
            return None
        ckpts = glob.glob(os.path.join(self._checkpoint_dir, "checkpoint_*.pt"))
        if not ckpts:
            return None
        best = None
        best_loss = float("inf")
        for path in ckpts:
            try:
                state = torch.load(path, map_location=self.device)
                val_loss = float(state["val_loss"])
                if val_loss < best_loss:
                    best_loss = val_loss
                    best = path
            except Exception:
                continue
        return best

    def train(self, train_loader, val_loader) -> None:
        print(f"开始训练，设备: {self.device}")
        print(f"训练轮数: {self.config.epochs}, 学习率: {self.config.learning_rate}")
        print("-" * 60)

        os.makedirs(self.config.output_dir, exist_ok=True)

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            print(
                f"Epoch [{epoch:2d}/{self.config.epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc*100:.2f}%"
            )

            self._save_checkpoint(epoch, val_loss, val_acc)

            if self._should_early_stop(val_loss):
                print(
                    f"Early stopping: 连续 {self.config.early_stopping_patience} 个 epoch 验证 loss 未下降"
                )
                break

        print("-" * 60)
        print("训练完成!")

        # 自动加载最优 checkpoint
        best_ckpt_path = self._find_best_checkpoint()
        if best_ckpt_path is not None:
            state = torch.load(best_ckpt_path, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])
            print(
                f"已加载最优模型 (epoch {state['epoch']}) 至 {self.config.model_path}"
            )

    def evaluate(self, test_loader) -> float:
        _, accuracy = self.validate(test_loader)
        print(f"测试集准确率: {accuracy*100:.2f}%")
        return accuracy

    def save_model(self, path: Optional[str] = None) -> str:
        path = path or self.config.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存至: {path}")
        return path

    def load_model(self, path: Optional[str] = None) -> None:
        path = path or self.config.model_path
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"模型已从 {path} 加载")

    def run_visualization(self, test_loader=None, y_true=None, y_pred=None, y_score=None):
        visualizer = Visualizer(self.config.output_dir)
        if self.config.save_loss_curve:
            visualizer.plot_losses(self.train_losses, self.val_losses)
        if test_loader is not None:
            y_true, y_pred, y_score = self._collect_predictions(test_loader)
        if self.config.save_roc_curve and y_score is not None:
            visualizer.plot_roc(y_true, y_score)
        if self.config.save_confusion_matrix and y_pred is not None:
            visualizer.plot_confusion_matrix(y_true, y_pred)

    def _collect_predictions(self, loader):
        self.model.eval()
        all_true = []
        all_pred = []
        all_score = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                proba = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                all_true.extend(labels.cpu().numpy())
                all_pred.extend(predicted.cpu().numpy())
                all_score.extend(proba.cpu().numpy())
        import numpy as np

        return (
            np.array(all_true),
            np.array(all_pred),
            np.array(all_score),
        )


def main():
    config = TrainConfig.from_cli()
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 60)
    print("步骤1: 数据加载与预处理")
    print("=" * 60)
    data_loader = URLDataLoader(config.data_path, max_features=config.max_features)
    train_loader, val_loader, test_loader, input_dim = data_loader.get_data_loaders(
        batch_size=config.batch_size
    )

    print("\n" + "=" * 60)
    print("步骤2: 模型构建")
    print("=" * 60)
    trainer = Trainer(config)
    trainer.build_model(input_dim=input_dim)
    print(trainer.model)
    print(f"模型参数量: {sum(p.numel() for p in trainer.model.parameters()):,}")

    print("\n" + "=" * 60)
    print("步骤3: 模型训练")
    print("=" * 60)
    trainer.train(train_loader, val_loader)

    print("\n" + "=" * 60)
    print("步骤4: 模型评估")
    print("=" * 60)
    test_accuracy = trainer.evaluate(test_loader)

    print("\n" + "=" * 60)
    print("步骤5: 可视化与保存")
    print("=" * 60)
    trainer.run_visualization(test_loader=test_loader)
    trainer.save_model(config.model_path)

    print("\n" + "=" * 60)
    print("模型分析")
    print("=" * 60)
    print(f"最终验证准确率: {trainer.val_accuracies[-1]*100:.2f}%")
    print(f"测试集准确率: {test_accuracy*100:.2f}%")
    print(f"最低验证损失: {min(trainer.val_losses):.4f}")
    return trainer


if __name__ == "__main__":
    main()
