"""
训练配置模块
使用 dataclass 定义训练配置，支持 YAML 文件和命令行参数两种方式加载
"""
import argparse
import os
from dataclasses import dataclass, field, fields, asdict
from typing import Optional


@dataclass
class TrainConfig:
    data_path: str = "data/url_dataset.csv"
    batch_size: int = 32
    learning_rate: float = 5e-5
    epochs: int = 25
    hidden_dim: int = 128
    max_features: int = 1200
    output_dir: str = "output"
    early_stopping_patience: int = 5
    checkpoint_keep_best: int = 3
    seed: int = 42
    device: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainConfig":
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            return cls()
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_args(cls, args=None) -> "TrainConfig":
        parser = argparse.ArgumentParser(description="恶意URL分类模型训练")
        parser.add_argument("--config", type=str, default=None, help="YAML配置文件路径")
        parser.add_argument("--data_path", type=str, default=None, help="数据集文件路径")
        parser.add_argument("--batch_size", type=int, default=None, help="批次大小")
        parser.add_argument("--learning_rate", type=float, default=None, help="学习率")
        parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
        parser.add_argument("--hidden_dim", type=int, default=None, help="隐藏层维度")
        parser.add_argument("--max_features", type=int, default=None, help="TF-IDF最大特征数")
        parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
        parser.add_argument("--early_stopping_patience", type=int, default=None, help="Early stopping耐心值")
        parser.add_argument("--checkpoint_keep_best", type=int, default=None, help="保留最优checkpoint数量")
        parser.add_argument("--seed", type=int, default=None, help="随机种子")
        parser.add_argument("--device", type=str, default=None, help="计算设备")
        parsed = parser.parse_args(args)

        if parsed.config:
            cfg = cls.from_yaml(parsed.config)
        else:
            cfg = cls()

        for f in fields(cls):
            val = getattr(parsed, f.name, None)
            if val is not None:
                setattr(cfg, f.name, val)

        return cfg

    def to_dict(self) -> dict:
        return asdict(self)
