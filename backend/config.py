"""
训练配置模块
使用 dataclass 定义训练参数，支持从 YAML 文件和命令行参数加载
"""
import argparse
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List


@dataclass
class TrainConfig:
    data_path: str = "data/url_dataset.csv"
    output_dir: str = "output"
    model_path: str = "output/best_model.pt"

    batch_size: int = 32
    learning_rate: float = 5e-5
    epochs: int = 25
    hidden_dim: int = 128
    max_features: int = 1200

    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    random_state: int = 42

    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    max_checkpoints: int = 3
    num_classes: int = 2

    device: str = ""
    save_loss_curve: bool = True
    save_roc_curve: bool = True
    save_confusion_matrix: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainConfig":
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_cli(cls, argv: Optional[List[str]] = None) -> "TrainConfig":
        parser = _build_argparser()
        args = parser.parse_args(argv)
        config = cls()
        for key, value in vars(args).items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)

        if args.config:
            yaml_config = cls.from_yaml(args.config)
            for key in cls.__dataclass_fields__:
                if getattr(config, key) == getattr(cls(), key):
                    setattr(config, key, getattr(yaml_config, key))

        if not config.device:
            import torch
            config.device = "cuda" if torch.cuda.is_available() else "cpu"
        return config


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="恶意URL分类模型训练")
    parser.add_argument("--config", type=str, default=None, help="YAML配置文件路径")

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--max_features", type=int, default=None)

    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--test_ratio", type=float, default=None)
    parser.add_argument("--random_state", type=int, default=None)

    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--early_stopping_min_delta", type=float, default=None)
    parser.add_argument("--max_checkpoints", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=None)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_loss_curve", type=int, default=None)
    parser.add_argument("--save_roc_curve", type=int, default=None)
    parser.add_argument("--save_confusion_matrix", type=int, default=None)
    return parser
