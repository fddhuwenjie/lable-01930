"""
Flask API服务
提供恶意URL分类的REST API接口
"""
import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

from model import URLClassifier
from data_loader import URLDataLoader

app = Flask(__name__)
CORS(app)

# 全局变量
model = None
vectorizer = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model():
    """加载训练好的模型"""
    global model, vectorizer
    
    model_path = "output/url_classifier.pth"
    data_path = "data/url_dataset.csv"
    
    # 初始化数据加载器以获取vectorizer
    data_loader = URLDataLoader(data_path, max_features=1200)
    data_loader.extract_features()
    vectorizer = data_loader.vectorizer
    
    # 获取实际特征维度
    input_dim = data_loader.features.shape[1]
    print(f"特征维度: {input_dim}")
    
    # 加载模型
    model = URLClassifier(input_dim=input_dim)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"模型已加载: {model_path}")
    else:
        print("警告: 未找到预训练模型，使用随机初始化权重")
    
    model.to(device)
    model.eval()


@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': device
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    URL分类预测接口
    
    请求体:
        {"url": "http://example.com"}
    
    返回:
        {"url": "...", "label": 0/1, "label_name": "正常/恶意", "confidence": 0.xx}
    """
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({'error': '请提供URL参数'}), 400
        
        url = data['url']
        
        # 特征提取
        features = vectorizer.transform([url]).toarray()
        features_tensor = torch.FloatTensor(features).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(features_tensor)
            proba = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        label = predicted.item()
        confidence = proba[0][label].item()
        
        return jsonify({
            'url': url,
            'label': label,
            'label_name': '恶意' if label == 1 else '正常',
            'confidence': round(confidence, 4)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    批量URL分类预测接口
    
    请求体:
        {"urls": ["http://example1.com", "http://example2.com"]}
    """
    try:
        data = request.get_json()
        
        if not data or 'urls' not in data:
            return jsonify({'error': '请提供urls参数'}), 400
        
        urls = data['urls']
        
        # 特征提取
        features = vectorizer.transform(urls).toarray()
        features_tensor = torch.FloatTensor(features).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(features_tensor)
            proba = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        results = []
        for i, url in enumerate(urls):
            label = predicted[i].item()
            confidence = proba[i][label].item()
            results.append({
                'url': url,
                'label': label,
                'label_name': '恶意' if label == 1 else '正常',
                'confidence': round(confidence, 4)
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train', methods=['POST'])
def train_model():
    """触发模型训练"""
    try:
        from trainer import main as train_main
        trainer = train_main(args=[])
        
        # 重新加载模型
        load_model()
        
        return jsonify({
            'status': 'success',
            'message': '模型训练完成',
            'final_val_accuracy': trainer.val_accuracies[-1]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
