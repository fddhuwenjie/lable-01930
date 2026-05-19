"""
训练配置模块
使用 dataclass 定义训练配置，支持从 YAML 文件和命令行参数加载
"""
from dataclasses import dataclass, field
from typing import Optional
import argparse
import yaml
import os


@dataclass
class TrainConfig:
    """
    训练配置类
    
    包含数据、模型、训练、输出等所有相关配置
    """
    data_path: str = "data/url_dataset.csv"
    batch_size: int = 32
    learning_rate: float = 5e-5
    epochs: int = 25
    hidden_dim: int = 128
    max_features: int = 1200
    early_stopping_patience: int = 5
    max_checkpoints: int = 3
    output_dir: str = "output"
    random_state: int = 42
    device: Optional[str] = None
    config_path: Optional[str] = None

    @classmethod
    def from_yaml(cls, config_path: str) -> "TrainConfig":
        """
        从 YAML 文件加载配置
        
        Args:
            config_path: YAML 配置文件路径
            
        Returns:
            TrainConfig 实例
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return cls(**filtered_config)

    @classmethod
    def from_args(cls) -> "TrainConfig":
        """
        从命令行参数加载配置
        
        Returns:
            TrainConfig 实例
        """
        parser = argparse.ArgumentParser(description="恶意URL分类模型训练")
        
        parser.add_argument(
            "--config", 
            type=str, 
            default=None,
            help="YAML 配置文件路径"
        )
        parser.add_argument(
            "--data-path", 
            type=str, 
            default=None,
            help="数据集文件路径"
        )
        parser.add_argument(
            "--batch-size", 
            type=int, 
            default=None,
            help="批次大小"
        )
        parser.add_argument(
            "--learning-rate", 
            type=float, 
            default=None,
            help="学习率"
        )
        parser.add_argument(
            "--epochs", 
            type=int, 
            default=None,
            help="训练轮数"
        )
        parser.add_argument(
            "--hidden-dim", 
            type=int, 
            default=None,
            help="隐藏层维度"
        )
        parser.add_argument(
            "--max-features", 
            type=int, 
            default=None,
            help="TF-IDF 最大特征数"
        )
        parser.add_argument(
            "--early-stopping-patience", 
            type=int, 
            default=None,
            help="早停耐心值（连续多少个 epoch 验证 loss 不下降则停止）"
        )
        parser.add_argument(
            "--max-checkpoints", 
            type=int, 
            default=None,
            help="最多保留的 checkpoint 数量"
        )
        parser.add_argument(
            "--output-dir", 
            type=str, 
            default=None,
            help="输出目录"
        )
        parser.add_argument(
            "--random-state", 
            type=int, 
            default=None,
            help="随机种子"
        )
        parser.add_argument(
            "--device", 
            type=str, 
            default=None,
            help="计算设备 (cpu/cuda)"
        )
        
        args = parser.parse_args()
        
        config = cls()
        
        if args.config:
            config = cls.from_yaml(args.config)
            config.config_path = args.config
        
        if args.data_path:
            config.data_path = args.data_path
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.learning_rate:
            config.learning_rate = args.learning_rate
        if args.epochs:
            config.epochs = args.epochs
        if args.hidden_dim:
            config.hidden_dim = args.hidden_dim
        if args.max_features:
            config.max_features = args.max_features
        if args.early_stopping_patience:
            config.early_stopping_patience = args.early_stopping_patience
        if args.max_checkpoints:
            config.max_checkpoints = args.max_checkpoints
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.random_state:
            config.random_state = args.random_state
        if args.device:
            config.device = args.device
        
        return config

    def save_to_yaml(self, save_path: str) -> None:
        """
        将配置保存为 YAML 文件
        
        Args:
            save_path: 保存路径
        """
        config_dict = {
            "data_path": self.data_path,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "hidden_dim": self.hidden_dim,
            "max_features": self.max_features,
            "early_stopping_patience": self.early_stopping_patience,
            "max_checkpoints": self.max_checkpoints,
            "output_dir": self.output_dir,
            "random_state": self.random_state,
            "device": self.device
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
