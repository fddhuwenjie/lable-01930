"""
恶意URL分类模型
基于PyTorch的全连接神经网络
"""
import torch
import torch.nn as nn


class URLClassifier(nn.Module):
    """
    恶意URL分类器
    包含1个隐藏层（128个神经元）的全连接神经网络
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 2):
        """
        初始化模型
        
        Args:
            input_dim: 输入特征维度（TF-IDF特征数）
            hidden_dim: 隐藏层神经元数量，默认128
            num_classes: 输出类别数，默认2（正常/恶意）
        """
        super(URLClassifier, self).__init__()
        
        # 全连接层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, input_dim)
            
        Returns:
            输出张量，形状为 (batch_size, num_classes)
            注意：输出层无激活函数，配合CrossEntropyLoss使用
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测类别
        
        Args:
            x: 输入张量
            
        Returns:
            预测的类别标签
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
        return predicted
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测概率
        
        Args:
            x: 输入张量
            
        Returns:
            各类别的概率
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            proba = torch.softmax(outputs, dim=1)
        return proba


if __name__ == "__main__":
    # 测试模型
    model = URLClassifier(input_dim=1200)
    print(model)
    
    # 测试前向传播
    x = torch.randn(32, 1200)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
