"""
URL数据加载器 - 参考SpamDataLoader类实现
用于加载恶意URL数据集
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader


class URLDataLoader:
    """
    恶意URL数据加载器
    负责数据读取、TF-IDF特征提取、数据集划分
    """
    
    def __init__(self, data_path: str, max_features: int = 1200):
        """
        初始化数据加载器
        
        Args:
            data_path: 数据集文件路径
            max_features: TF-IDF最大特征数，默认1200
        """
        self.data_path = data_path
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            token_pattern=r'[a-zA-Z0-9]+',
            lowercase=True
        )
        self.data = None
        self.features = None
        self.labels = None
        
    def load_data(self) -> pd.DataFrame:
        """
        读取CSV数据文件
        
        Returns:
            包含URL和标签的DataFrame
        """
        self.data = pd.read_csv(self.data_path)
        print(f"数据加载完成，共 {len(self.data)} 条记录")
        print(f"正常URL: {len(self.data[self.data['label'] == 0])} 条")
        print(f"恶意URL: {len(self.data[self.data['label'] == 1])} 条")
        return self.data
    
    def extract_features(self) -> np.ndarray:
        """
        使用TF-IDF提取URL文本特征
        
        Returns:
            TF-IDF特征矩阵
        """
        if self.data is None:
            self.load_data()
        
        urls = self.data['url'].values
        self.features = self.vectorizer.fit_transform(urls).toarray()
        self.labels = self.data['label'].values
        
        print(f"特征提取完成，特征维度: {self.features.shape}")
        return self.features
    
    def split_data(self, train_ratio: float = 0.7, val_ratio: float = 0.2, 
                   test_ratio: float = 0.1, random_state: int = 42):
        """
        划分训练集、验证集、测试集
        
        Args:
            train_ratio: 训练集比例，默认70%
            val_ratio: 验证集比例，默认20%
            test_ratio: 测试集比例，默认10%
            random_state: 随机种子
            
        Returns:
            训练集、验证集、测试集的特征和标签元组
        """
        if self.features is None:
            self.extract_features()
        
        # 先划分出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.features, self.labels,
            test_size=test_ratio,
            random_state=random_state,
            stratify=self.labels
        )
        
        # 从剩余数据中划分训练集和验证集
        val_size = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=random_state,
            stratify=y_temp
        )
        
        print(f"数据集划分完成:")
        print(f"  训练集: {len(X_train)} 条 ({train_ratio*100:.0f}%)")
        print(f"  验证集: {len(X_val)} 条 ({val_ratio*100:.0f}%)")
        print(f"  测试集: {len(X_test)} 条 ({test_ratio*100:.0f}%)")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def to_tensors(self, train_data, val_data, test_data):
        """
        将数据转换为PyTorch张量
        
        Args:
            train_data: 训练数据元组 (X_train, y_train)
            val_data: 验证数据元组 (X_val, y_val)
            test_data: 测试数据元组 (X_test, y_test)
            
        Returns:
            PyTorch张量格式的数据
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        return (
            (X_train_tensor, y_train_tensor),
            (X_val_tensor, y_val_tensor),
            (X_test_tensor, y_test_tensor)
        )
    
    def get_data_loaders(self, batch_size: int = 32):
        """
        获取PyTorch DataLoader
        
        Args:
            batch_size: 批次大小，默认32
            
        Returns:
            训练、验证、测试的DataLoader
        """
        # 加载和处理数据
        train_data, val_data, test_data = self.split_data()
        tensors = self.to_tensors(train_data, val_data, test_data)
        
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = tensors
        
        # 创建TensorDataset
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        # 创建DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, self.features.shape[1]


if __name__ == "__main__":
    # 测试数据加载器
    loader = URLDataLoader("data/url_dataset.csv")
    train_loader, val_loader, test_loader, input_dim = loader.get_data_loaders()
    print(f"\n输入特征维度: {input_dim}")
