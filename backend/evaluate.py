"""
独立评估脚本
支持加载已保存的模型对新数据进行批量预测
用法: python evaluate.py --model output/best_model.pt --input data/test.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from data_loader import URLDataLoader
from model import URLClassifier
from visualizer import Visualizer


def load_model_for_eval(model_path: str, device: str):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        hidden_dim = checkpoint.get('hidden_dim', 128)
        max_features = checkpoint.get('max_features', 1200)
        state_dict = checkpoint['model_state_dict']
    else:
        hidden_dim = 128
        max_features = 1200
        state_dict = checkpoint

    fc1_weight = state_dict['fc1.weight']
    input_dim = fc1_weight.shape[1]
    hidden_dim_actual = fc1_weight.shape[0]
    if hidden_dim is None:
        hidden_dim = hidden_dim_actual
    model = URLClassifier(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, max_features


def predict_batch(model, data_loader, device: str):
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 2:
                inputs, labels = batch
            else:
                inputs = batch[0]
                labels = None

            inputs = inputs.to(device)
            outputs = model(inputs)
            proba = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_probs.extend(proba[:, 1].cpu().numpy().tolist())

            if labels is not None:
                all_labels.extend(labels.numpy().tolist())

    return all_labels, all_preds, all_probs


def main():
    parser = argparse.ArgumentParser(description="恶意URL分类模型独立评估")
    parser.add_argument("--model", type=str, required=True, help="模型文件路径 (如 output/best_model.pt)")
    parser.add_argument("--input", type=str, required=True, help="待评估的CSV数据文件路径")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--output_dir", type=str, default="output/eval", help="评估结果输出目录")
    parser.add_argument("--train_data", type=str, default="data/url_dataset.csv",
                        help="训练数据路径（用于拟合 TF-IDF vectorizer）")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"错误: 数据文件不存在: {args.input}")
        sys.exit(1)

    print("=" * 60)
    print("加载模型")
    print("=" * 60)

    model, max_features = load_model_for_eval(args.model, device)
    print(f"模型已从 {args.model} 加载")
    print(f"模型 max_features: {max_features}")

    print("\n" + "=" * 60)
    print("数据预处理")
    print("=" * 60)

    train_loader_obj = URLDataLoader(args.train_data, max_features=max_features)
    train_loader_obj.extract_features()
    vectorizer = train_loader_obj.vectorizer

    df = pd.read_csv(args.input)
    print(f"数据加载完成，共 {len(df)} 条记录")

    urls = df['url'].values
    features = vectorizer.transform(urls).toarray()

    X_tensor = torch.FloatTensor(features)

    has_labels = 'label' in df.columns
    if has_labels:
        y_tensor = torch.LongTensor(df['label'].values)
        dataset = TensorDataset(X_tensor, y_tensor)
    else:
        y_tensor = torch.zeros(len(df), dtype=torch.LongTensor)
        dataset = TensorDataset(X_tensor, y_tensor)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print("\n" + "=" * 60)
    print("批量预测")
    print("=" * 60)

    all_labels, all_preds, all_probs = predict_batch(model, data_loader, device)

    label_names = ['正常' if p == 0 else '恶意' for p in all_preds]
    results = pd.DataFrame({
        'url': urls,
        'predicted_label': all_preds,
        'predicted_name': label_names,
        'malicious_prob': [round(p, 4) for p in all_probs],
    })

    os.makedirs(args.output_dir, exist_ok=True)

    if has_labels:
        true_labels = df['label'].values.tolist()
        results['true_label'] = true_labels

        correct = sum(1 for t, p in zip(true_labels, all_preds) if t == p)
        accuracy = correct / len(true_labels)
        print(f"准确率: {accuracy * 100:.2f}%")

        visualizer = Visualizer()
        visualizer.plot_roc_curve(true_labels, all_probs,
                                  save_path=os.path.join(args.output_dir, "roc_curve.png"))
        visualizer.plot_confusion_matrix(true_labels, all_preds,
                                         save_path=os.path.join(args.output_dir, "confusion_matrix.png"))

    result_path = os.path.join(args.output_dir, "predictions.csv")
    results.to_csv(result_path, index=False, encoding='utf-8')
    print(f"\n预测结果已保存至: {result_path}")
    print(f"总预测样本数: {len(all_preds)}")
    print(f"恶意URL数: {sum(all_preds)}")
    print(f"正常URL数: {len(all_preds) - sum(all_preds)}")


if __name__ == "__main__":
    main()
