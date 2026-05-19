"""
模型评估脚本
支持加载已保存的模型对新数据进行批量预测
"""
import torch
import argparse
import os
import numpy as np
import pickle
from typing import Tuple, List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import TensorDataset, DataLoader

from model import URLClassifier
from visualizer import Visualizer


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """
    加载保存的 TF-IDF 向量化器
    
    Args:
        vectorizer_path: 向量化器文件路径
        
    Returns:
        TF-IDF 向量化器
    """
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"向量化器已从 {vectorizer_path} 加载")
    return vectorizer


def load_data_with_vectorizer(
    data_path: str,
    vectorizer: TfidfVectorizer,
    batch_size: int = 32
) -> Tuple[DataLoader, int]:
    """
    使用已有的向量化器加载数据
    
    Args:
        data_path: 数据文件路径
        vectorizer: 已训练的 TF-IDF 向量化器
        batch_size: 批次大小
        
    Returns:
        (数据加载器, 特征维度)
    """
    df = pd.read_csv(data_path)
    print(f"数据加载完成，共 {len(df)} 条记录")
    
    if 'label' in df.columns:
        print(f"正常URL: {len(df[df['label'] == 0])} 条")
        print(f"恶意URL: {len(df[df['label'] == 1])} 条")
    
    urls = df['url'].values
    features = vectorizer.transform(urls).toarray()
    print(f"特征提取完成，特征维度: {features.shape}")
    
    if 'label' in df.columns:
        labels = df['label'].values
    else:
        labels = np.zeros(len(df))
    
    X_tensor = torch.FloatTensor(np.ascontiguousarray(features).copy())
    y_tensor = torch.LongTensor(np.ascontiguousarray(labels).copy())
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return loader, features.shape[1]


def evaluate_model(
    model_path: str,
    input_path: str,
    vectorizer_path: str = None,
    output_dir: str = "output",
    batch_size: int = 32,
    hidden_dim: int = 128,
    device: str = None
) -> dict:
    """
    评估模型
    
    Args:
        model_path: 模型文件路径
        input_path: 输入数据文件路径
        vectorizer_path: 向量化器文件路径（默认从模型同目录加载）
        output_dir: 输出目录
        batch_size: 批次大小
        hidden_dim: 隐藏层维度
        device: 计算设备
        
    Returns:
        评估结果字典
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    if vectorizer_path is None:
        vectorizer_path = os.path.join(os.path.dirname(model_path), "vectorizer.pkl")
    
    print("=" * 60)
    print("模型评估")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"输入数据: {input_path}")
    print(f"向量化器: {vectorizer_path}")
    print(f"设备: {device}")
    print()
    
    vectorizer = load_vectorizer(vectorizer_path)
    
    test_loader, input_dim = load_data_with_vectorizer(
        input_path,
        vectorizer,
        batch_size=batch_size
    )
    
    print("\n" + "-" * 60)
    print("加载模型")
    model = URLClassifier(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"模型已从 {model_path} 加载")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n" + "-" * 60)
    print("开始评估")
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    all_urls = []
    correct = 0
    total = 0
    
    df = pd.read_csv(input_path)
    urls = df['url'].values
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            batch_start = i * batch_size
            batch_end = min(batch_start + batch_size, len(urls))
            all_urls.extend(urls[batch_start:batch_end])
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = correct / total
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)
    
    print(f"准确率: {accuracy*100:.2f}%")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results_df = pd.DataFrame({
        'url': all_urls,
        'true_label': y_true,
        'predicted_label': y_pred,
        'prob_benign': y_proba[:, 0],
        'prob_malicious': y_proba[:, 1]
    })
    
    results_path = os.path.join(output_dir, "evaluation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"预测结果已保存至: {results_path}")
    
    visualizer = Visualizer()
    
    roc_auc = visualizer.plot_roc_curve(
        y_true,
        y_proba[:, 1],
        save_path=os.path.join(output_dir, "evaluation_roc.png")
    )
    print(f"ROC AUC: {roc_auc:.4f}")
    
    cm = visualizer.plot_confusion_matrix(
        y_true,
        y_pred,
        save_path=os.path.join(output_dir, "evaluation_confusion_matrix.png")
    )
    
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "=" * 60)
    print("评估指标")
    print("=" * 60)
    print(f"准确率 (Accuracy): {accuracy*100:.2f}%")
    print(f"精确率 (Precision): {precision*100:.2f}%")
    print(f"召回率 (Recall): {recall*100:.2f}%")
    print(f"F1 分数: {f1*100:.2f}%")
    print(f"ROC AUC: {roc_auc:.4f}")
    print()
    print("混淆矩阵:")
    print(f"  真阴性 (TN): {tn}")
    print(f"  假阳性 (FP): {fp}")
    print(f"  假阴性 (FN): {fn}")
    print(f"  真阳性 (TP): {tp}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'predictions': results_df
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="恶意URL分类模型评估")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="已训练的模型文件路径 (如 output/best_model.pt)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="待评估的 CSV 数据文件路径"
    )
    parser.add_argument(
        "--vectorizer",
        type=str,
        default=None,
        help="TF-IDF 向量化器文件路径（默认从模型同目录加载 vectorizer.pkl）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="评估结果输出目录"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批次大小"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="模型隐藏层维度"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备 (cpu/cuda)"
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        input_path=args.input,
        vectorizer_path=args.vectorizer,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        device=args.device
    )


if __name__ == "__main__":
    main()
