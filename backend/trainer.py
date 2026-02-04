"""
模型训练器
负责模型训练、验证和评估
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
import os

from data_loader import URLDataLoader
from model import URLClassifier


class Trainer:
    """
    模型训练器
    """
    
    def __init__(self, model: nn.Module, learning_rate: float = 5e-5, device: str = None):
        """
        初始化训练器
        
        Args:
            model: PyTorch模型
            learning_rate: 学习率，默认5e-5
            device: 计算设备
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 记录训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader) -> float:
        """
        训练一个epoch
        
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
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
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
    
    def train(self, train_loader, val_loader, epochs: int = 25) -> None:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数，默认25
        """
        print(f"开始训练，设备: {self.device}")
        print(f"训练轮数: {epochs}, 学习率: {self.optimizer.param_groups[0]['lr']}")
        print("-" * 60)
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"Epoch [{epoch+1:2d}/{epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc*100:.2f}%")
        
        print("-" * 60)
        print("训练完成!")
        
    def evaluate(self, test_loader) -> float:
        """
        在测试集上评估模型
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            测试集准确率
        """
        _, accuracy = self.validate(test_loader)
        print(f"测试集准确率: {accuracy*100:.2f}%")
        return accuracy
    
    def plot_losses(self, save_path: str = None) -> None:
        """
        绘制训练/验证损失曲线
        
        Args:
            save_path: 图片保存路径
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.plot(epochs, self.train_losses, 'b-', label='训练损失', linewidth=2)
        plt.plot(epochs, self.val_losses, 'r-', label='验证损失', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('训练/验证损失曲线', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"损失曲线已保存至: {save_path}")
        
        plt.close()
    
    def save_model(self, path: str) -> None:
        """保存模型"""
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存至: {path}")
    
    def load_model(self, path: str) -> None:
        """加载模型"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"模型已从 {path} 加载")


def main():
    """主函数：执行完整的训练流程"""
    # 配置参数
    DATA_PATH = "data/url_dataset.csv"
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-5
    EPOCHS = 25
    HIDDEN_DIM = 128
    MAX_FEATURES = 1200
    
    # 创建输出目录
    os.makedirs("output", exist_ok=True)
    
    # 1. 数据加载
    print("=" * 60)
    print("步骤1: 数据加载与预处理")
    print("=" * 60)
    
    data_loader = URLDataLoader(DATA_PATH, max_features=MAX_FEATURES)
    train_loader, val_loader, test_loader, input_dim = data_loader.get_data_loaders(
        batch_size=BATCH_SIZE
    )
    
    # 2. 模型构建
    print("\n" + "=" * 60)
    print("步骤2: 模型构建")
    print("=" * 60)
    
    model = URLClassifier(input_dim=input_dim, hidden_dim=HIDDEN_DIM)
    print(model)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 模型训练
    print("\n" + "=" * 60)
    print("步骤3: 模型训练")
    print("=" * 60)
    
    trainer = Trainer(model, learning_rate=LEARNING_RATE)
    trainer.train(train_loader, val_loader, epochs=EPOCHS)
    
    # 4. 模型评估
    print("\n" + "=" * 60)
    print("步骤4: 模型评估")
    print("=" * 60)
    
    test_accuracy = trainer.evaluate(test_loader)
    
    # 5. 绘制损失曲线
    print("\n" + "=" * 60)
    print("步骤5: 绘制损失曲线")
    print("=" * 60)
    
    trainer.plot_losses(save_path="output/loss_curve.png")
    
    # 6. 保存模型
    trainer.save_model("output/url_classifier.pth")
    
    # 7. 分析结果
    print("\n" + "=" * 60)
    print("模型分析")
    print("=" * 60)
    print(f"最终验证准确率: {trainer.val_accuracies[-1]*100:.2f}%")
    print(f"测试集准确率: {test_accuracy*100:.2f}%")
    print(f"最低验证损失: {min(trainer.val_losses):.4f}")
    
    return trainer


if __name__ == "__main__":
    main()
