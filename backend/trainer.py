"""
模型训练器
负责模型训练、验证、评估、保存和加载
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
from typing import Tuple, List, Dict, Optional
from collections import deque
import glob

from data_loader import URLDataLoader
from model import URLClassifier
from config import TrainConfig
from visualizer import Visualizer


class Trainer:
    """
    模型训练器
    """
    
    def __init__(self, config: TrainConfig):
        """
        初始化训练器
        
        Args:
            config: 训练配置
        """
        self.config = config
        self.device = config.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []
        
        self.best_val_loss: float = float('inf')
        self.best_epoch: int = -1
        self.checkpoint_manager = CheckpointManager(
            output_dir=config.output_dir,
            max_checkpoints=config.max_checkpoints
        )
        
        self.model: Optional[nn.Module] = None
        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        
    def setup_model(self, input_dim: int) -> None:
        """
        初始化模型、损失函数和优化器
        
        Args:
            input_dim: 输入特征维度
        """
        self.model = URLClassifier(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        print(f"模型已初始化，输入维度: {input_dim}, 隐藏层维度: {self.config.hidden_dim}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader) -> float:
        """
        训练一个 epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            平均训练损失
        """
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
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            验证损失和准确率
        """
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
    
    def train(self, train_loader, val_loader) -> None:
        """
        完整训练流程（包含 early stopping）
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        print(f"开始训练，设备: {self.device}")
        print(f"训练轮数: {self.config.epochs}, 学习率: {self.config.learning_rate}")
        print(f"早停耐心值: {self.config.early_stopping_patience}")
        print("-" * 60)
        
        epochs_without_improvement = 0
        
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
            
            checkpoint_path = os.path.join(
                self.config.output_dir,
                f"checkpoint_epoch_{epoch+1:03d}_loss_{val_loss:.4f}.pt"
            )
            self.save_model(checkpoint_path)
            self.checkpoint_manager.add_checkpoint(checkpoint_path, val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                epochs_without_improvement = 0
                
                best_model_path = os.path.join(self.config.output_dir, "best_model.pt")
                self.save_model(best_model_path)
                print(f"  -> 新的最优模型！验证损失: {val_loss:.4f}")
            else:
                epochs_without_improvement += 1
                print(f"  -> 无改善，耐心值: {epochs_without_improvement}/{self.config.early_stopping_patience}")
            
            if epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"\n早停触发！连续 {self.config.early_stopping_patience} 个 epoch 验证损失未下降")
                break
        
        print("-" * 60)
        print(f"训练完成！最优模型在 Epoch {self.best_epoch}，验证损失: {self.best_val_loss:.4f}")
    
    def evaluate(self, test_loader) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        在测试集上评估模型
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            (准确率, 真实标签, 预测标签, 预测概率)
        """
        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        accuracy = correct / total
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_proba = np.array(all_probs)
        
        print(f"测试集准确率: {accuracy*100:.2f}%")
        
        return accuracy, y_true, y_pred, y_proba
    
    def save_model(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str) -> None:
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"模型已从 {path} 加载")


class CheckpointManager:
    """
    Checkpoint 管理器，负责保留最优的 N 个 checkpoint
    """
    
    def __init__(self, output_dir: str, max_checkpoints: int = 3):
        """
        初始化 Checkpoint 管理器
        
        Args:
            output_dir: 输出目录
            max_checkpoints: 最多保留的 checkpoint 数量
        """
        self.output_dir = output_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[Tuple[str, float]] = []
        
    def add_checkpoint(self, path: str, loss: float) -> None:
        """
        添加一个 checkpoint，超过最大数量时删除最差的
        
        Args:
            path: checkpoint 文件路径
            loss: 验证损失（越小越好）
        """
        self.checkpoints.append((path, loss))
        self.checkpoints.sort(key=lambda x: x[1])
        
        while len(self.checkpoints) > self.max_checkpoints:
            removed = self.checkpoints.pop()
            if os.path.exists(removed[0]):
                os.remove(removed[0])
                print(f"  -> 删除旧 checkpoint: {os.path.basename(removed[0])}")
    
    def get_best_checkpoint(self) -> Optional[str]:
        """
        获取最优的 checkpoint 路径
        
        Returns:
            最优 checkpoint 路径，若无则返回 None
        """
        if not self.checkpoints:
            return None
        return self.checkpoints[0][0]


def main():
    """主函数：执行完整的训练流程"""
    config = TrainConfig.from_args()
    
    os.makedirs(config.output_dir, exist_ok=True)
    config.save_to_yaml(os.path.join(config.output_dir, "config.yaml"))
    
    print("=" * 60)
    print("步骤1: 数据加载与预处理")
    print("=" * 60)
    
    data_loader = URLDataLoader(config.data_path, max_features=config.max_features)
    train_loader, val_loader, test_loader, input_dim = data_loader.get_data_loaders(
        batch_size=config.batch_size
    )
    
    vectorizer_path = os.path.join(config.output_dir, "vectorizer.pkl")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(data_loader.vectorizer, f)
    print(f"向量化器已保存至: {vectorizer_path}")
    
    print("\n" + "=" * 60)
    print("步骤2: 模型构建")
    print("=" * 60)
    
    trainer = Trainer(config)
    trainer.setup_model(input_dim)
    
    print("\n" + "=" * 60)
    print("步骤3: 模型训练")
    print("=" * 60)
    
    trainer.train(train_loader, val_loader)
    
    print("\n" + "=" * 60)
    print("步骤4: 模型评估")
    print("=" * 60)
    
    best_model_path = os.path.join(config.output_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        trainer.load_model(best_model_path)
    
    test_accuracy, y_true, y_pred, y_proba = trainer.evaluate(test_loader)
    
    print("\n" + "=" * 60)
    print("步骤5: 可视化")
    print("=" * 60)
    
    visualizer = Visualizer()
    visualizer.plot_losses(
        trainer.train_losses,
        trainer.val_losses,
        save_path=os.path.join(config.output_dir, "loss_curve.png")
    )
    
    visualizer.plot_roc_curve(
        y_true,
        y_proba[:, 1],
        save_path=os.path.join(config.output_dir, "roc_curve.png")
    )
    
    visualizer.plot_confusion_matrix(
        y_true,
        y_pred,
        save_path=os.path.join(config.output_dir, "confusion_matrix.png")
    )
    
    print("\n" + "=" * 60)
    print("模型分析")
    print("=" * 60)
    print(f"最终验证准确率: {trainer.val_accuracies[-1]*100:.2f}%")
    print(f"测试集准确率: {test_accuracy*100:.2f}%")
    print(f"最低验证损失: {min(trainer.val_losses):.4f}")
    print(f"最优模型: {os.path.join(config.output_dir, 'best_model.pt')}")
    
    return trainer


if __name__ == "__main__":
    main()
