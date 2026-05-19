"""
独立评估脚本
加载已保存模型对新数据进行批量预测
用法: python evaluate.py --model output/best_model.pt --input data/test.csv
"""
import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data_loader import URLDataLoader
from model import URLClassifier
from config import TrainConfig
from visualizer import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description="恶意URL分类模型独立评估")
    parser.add_argument("--model", type=str, required=True, help="已保存模型权重路径 (.pt/.pth)")
    parser.add_argument("--input", type=str, required=True, help="待评估数据CSV文件路径 (含 url 和 label 列)")
    parser.add_argument("--output", type=str, default="output", help="评估结果输出目录")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--max_features", type=int, default=1200)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="限制评估样本数量，用于快速测试")
    parser.add_argument("--save_predictions", type=int, default=1, help="是否保存预测结果CSV (1/0)")
    return parser.parse_args()


def build_inference_loader(data_loader: URLDataLoader, csv_path: str, batch_size: int, max_samples: int = None):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("输入CSV必须包含 'label' 列")
    if "url" not in df.columns:
        raise ValueError("输入CSV必须包含 'url' 列")

    if hasattr(data_loader, "vectorizer") and data_loader.vectorizer is not None and hasattr(data_loader.vectorizer, "vocabulary_"):
        X = data_loader.vectorizer.transform(df["url"].values).toarray()
    else:
        data_loader.data = df
        X = data_loader.vectorizer.fit_transform(df["url"].values).toarray()

    y = df["label"].values
    if max_samples is not None:
        X = X[:max_samples]
        y = y[:max_samples]

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, X.shape[1]


def collect_predictions(model, loader, device):
    model.eval()
    all_true = []
    all_pred = []
    all_score = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            proba = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())
            all_score.extend(proba.cpu().numpy())
    return np.array(all_true), np.array(all_pred), np.array(all_score)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("步骤1: 加载数据与向量化器")
    print("=" * 60)
    data_loader = URLDataLoader(args.input, max_features=args.max_features)
    loader, input_dim = build_inference_loader(
        data_loader, args.input, args.batch_size, args.max_samples
    )
    print(f"输入特征维度: {input_dim}")

    print("\n" + "=" * 60)
    print("步骤2: 加载模型权重")
    print("=" * 60)
    model = URLClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=2,
    ).to(device)

    state = torch.load(args.model, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict):
        try:
            model.load_state_dict(state)
        except Exception:
            raise ValueError(
                f"无法从 {args.model} 加载模型权重，期望的键: model_state_dict 或直接 state_dict"
            )
    else:
        raise ValueError(f"无法从 {args.model} 加载模型权重")
    print(f"模型已加载，设备: {device}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 60)
    print("步骤3: 批量预测")
    print("=" * 60)
    y_true, y_pred, y_score = collect_predictions(model, loader, device)

    accuracy = (y_pred == y_true).mean()
    print(f"评估样本数: {len(y_true)}")
    print(f"准确率: {accuracy*100:.2f}%")

    print("\n" + "=" * 60)
    print("步骤4: 可视化结果")
    print("=" * 60)
    visualizer = Visualizer(args.output)
    visualizer.plot_roc(y_true, y_score)
    visualizer.plot_confusion_matrix(y_true, y_pred)

    if args.save_predictions:
        df = pd.DataFrame(
            {
                "y_true": y_true,
                "y_pred": y_pred,
                "y_score_class0": y_score[:, 0],
                "y_score_class1": y_score[:, 1],
            }
        )
        pred_path = os.path.join(args.output, "predictions.csv")
        df.to_csv(pred_path, index=False)
        print(f"预测结果已保存至: {pred_path}")

    metrics = {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy),
        "model_path": args.model,
        "input_path": args.input,
    }
    metrics_path = os.path.join(args.output, "eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"评估指标已保存至: {metrics_path}")
    return metrics


if __name__ == "__main__":
    main()
