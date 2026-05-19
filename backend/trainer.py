"""
模型训练器
负责模型训练、验证和评估
支持 early stopping 和 checkpoint 保存
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
from typing import Tuple, List, Optional

from config import TrainConfig
from data_loader import URLDataLoader
from model import URLClassifier
from visualizer import Visualizer


class Trainer:
    def __init__(self, model: nn.Module, config: TrainConfig):
        self.config = config
        self.device = config.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []
        self.best_val_loss: float = float('inf')
        self.patience_counter: int = 0
        self.best_model_path: Optional[str] = None
        self._checkpoint_paths: List[Tuple[float, str]] = []

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

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        ckpt_path = os.path.join(
            self.config.output_dir, f"checkpoint_epoch_{epoch + 1}_valloss_{val_loss:.4f}.pt"
        )
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'hidden_dim': self.config.hidden_dim,
            'max_features': self.config.max_features,
        }, ckpt_path)
        self._checkpoint_paths.append((val_loss, ckpt_path))
        self._checkpoint_paths.sort(key=lambda x: x[0])

        while len(self._checkpoint_paths) > self.config.checkpoint_keep_best:
            _, path_to_remove = self._checkpoint_paths.pop()
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        if val_loss <= self._checkpoint_paths[0][0]:
            best_model_dst = os.path.join(self.config.output_dir, "best_model.pt")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'hidden_dim': self.config.hidden_dim,
                'max_features': self.config.max_features,
                'val_loss': val_loss,
            }, best_model_dst)
            self.best_model_path = best_model_dst

    def _collect_predictions(self, data_loader) -> Tuple[List[int], List[float], List[int]]:
        self.model.eval()
        all_labels: List[int] = []
        all_probs: List[float] = []
        all_preds: List[int] = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                proba = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy().tolist())
                all_probs.extend(proba[:, 1].cpu().numpy().tolist())
                all_preds.extend(predicted.cpu().numpy().tolist())

        return all_labels, all_probs, all_preds

    def train(self, train_loader, val_loader) -> None:
        print(f"开始训练，设备: {self.device}")
        print(f"训练轮数: {self.config.epochs}, 学习率: {self.config.learning_rate}")
        print(f"Early stopping patience: {self.config.early_stopping_patience}")
        print("-" * 60)

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            print(f"Epoch [{epoch+1:2d}/{self.config.epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc*100:.2f}%")

            self._save_checkpoint(epoch, val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"\nEarly stopping: 验证集 loss 连续 {self.config.early_stopping_patience} 个 epoch 未下降")
                    break

        print("-" * 60)
        print("训练完成!")

    def evaluate(self, test_loader) -> float:
        _, accuracy = self.validate(test_loader)
        print(f"测试集准确率: {accuracy*100:.2f}%")
        return accuracy

    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else self.config.output_dir, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'hidden_dim': self.config.hidden_dim,
            'max_features': self.config.max_features,
        }, path)
        print(f"模型已保存至: {path}")

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        print(f"模型已从 {path} 加载")


def main(args=None):
    config = TrainConfig.from_args(args)

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

    model = URLClassifier(input_dim=input_dim, hidden_dim=config.hidden_dim)
    print(model)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 60)
    print("步骤3: 模型训练")
    print("=" * 60)

    trainer = Trainer(model, config)
    trainer.train(train_loader, val_loader)

    print("\n" + "=" * 60)
    print("步骤4: 模型评估")
    print("=" * 60)

    test_accuracy = trainer.evaluate(test_loader)

    print("\n" + "=" * 60)
    print("步骤5: 绘制可视化图表")
    print("=" * 60)

    visualizer = Visualizer()
    visualizer.plot_loss_curve(trainer.train_losses, trainer.val_losses,
                               save_path=os.path.join(config.output_dir, "loss_curve.png"))

    all_labels, all_probs, all_preds = trainer._collect_predictions(test_loader)
    visualizer.plot_roc_curve(all_labels, all_probs,
                              save_path=os.path.join(config.output_dir, "roc_curve.png"))
    visualizer.plot_confusion_matrix(all_labels, all_preds,
                                     save_path=os.path.join(config.output_dir, "confusion_matrix.png"))

    print("\n" + "=" * 60)
    print("步骤6: 保存模型")
    print("=" * 60)

    trainer.save_model(os.path.join(config.output_dir, "url_classifier.pth"))

    print("\n" + "=" * 60)
    print("模型分析")
    print("=" * 60)
    print(f"最终验证准确率: {trainer.val_accuracies[-1]*100:.2f}%")
    print(f"测试集准确率: {test_accuracy*100:.2f}%")
    print(f"最低验证损失: {min(trainer.val_losses):.4f}")

    return trainer


if __name__ == "__main__":
    main()
