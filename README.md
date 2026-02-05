# 恶意URL分类系统

基于监督学习的恶意URL分类项目，使用PyTorch构建全连接神经网络，通过TF-IDF特征提取实现URL的二分类（正常/恶意）。

## How to Run

### Docker方式（推荐）

```bash
# 构建并启动服务
docker compose up --build -d

# 查看日志
docker compose logs -f

# 停止服务
docker compose down
```

### 本地运行

```bash
cd backend
pip install -r requirements.txt

# 训练模型
python trainer.py

# 启动API服务
python app.py
```

## Services

| 服务 | 端口 | 描述 |
|------|------|------|
| Backend API | 5001 | 恶意URL分类REST API服务 |

### API接口

- `GET /health` - 健康检查
- `POST /predict` - 单个URL预测
- `POST /batch_predict` - 批量URL预测
- `POST /train` - 触发模型训练

### 使用示例

```bash
# 健康检查
curl http://localhost:5001/health

# 单个URL预测
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "http://malware-download.com/virus.exe"}'

# 批量预测
curl -X POST http://localhost:5001/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"urls": ["http://google.com", "http://phishing-site.net/login"]}'
```

## 测试账号

本项目为机器学习分类服务，无需登录账号。

API服务启动后可直接调用接口进行URL分类预测。

## 题目内容

任务 1：基于监督学习的恶意 URL 分类 
数据预处理： 
自行查找并加载“恶意 URL 数据集”（包含 “URL 文本” 和 “标签”，标签为 “正常 = 0 / 恶意 = 1”），参考 SpamDataLoader 类实现 URLDataLoader ，完成数据读取； 
采用 TF-IDF 提取 URL 文本特征（设置最大特征数为 1200，过滤英文停用词），将特征与标签转换为 PyTorch 张量，划分训练集（70%）、验证集（20%）、测试集（10%）。 
模型构建： 
基于 PyTorch 继承 nn.Module 类，构建含 1 个隐藏层（128 个神经元）的全连接神经网络，隐藏层激活函数用 ReLU，输出层适配二分类（无需额外激活，配合交叉熵损失）。 
模型训练与评估： 
配置交叉熵损失函数（ nn.CrossEntropyLoss ），学习率设为 5e-5，训练轮数 25 轮，批次大小 32； 
每轮训练后计算验证集准确率，训练结束后计算测试集准确率，绘制 “训练 / 验证损失曲线”，分析模型对恶意 URL 的识别效果。
---


## 测试指南

### 1. 启动服务

```bash
# 清理旧容器并启动
docker compose down && docker compose up --build -d

# 等待服务启动（约60秒，需要训练模型）
docker compose logs -f
```

### 2. 健康检查

```bash
curl http://localhost:5001/health
```

预期返回：
```json
{"status": "healthy", "model_loaded": true, "device": "cpu"}
```

### 3. 单个URL预测测试

测试正常URL：
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.google.com"}'
```

测试恶意URL：
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "http://malware-download.com/virus.exe"}'
```

预期返回格式：
```json
{"url": "...", "label": 0, "label_name": "正常", "confidence": 0.95}
```

### 4. 批量预测测试

```bash
curl -X POST http://localhost:5001/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://www.github.com", "http://phishing-site.net/login.php", "https://www.amazon.com"]}'
```

### 5. 触发模型重新训练

```bash
curl -X POST http://localhost:5001/train
```

### 6. 错误处理测试

缺少参数：
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{}'
```

预期返回：
```json
{"error": "请提供URL参数"}
```

### 7. 常见问题排查

| 问题 | 解决方案 |
|------|----------|
| 容器名冲突 | `docker rm -f url-classifier-backend` |
| 端口被占用 | `lsof -i :5001` 查看占用进程 |
| 模型未加载 | 检查 `backend/output/` 目录是否有模型文件 |
| 服务无响应 | `docker compose logs` 查看日志 |


## 项目结构

```
.
├── backend/
│   ├── data/
│   │   └── url_dataset.csv      # 恶意URL数据集
│   ├── output/                   # 模型输出目录
│   ├── data_loader.py           # 数据加载器
│   ├── model.py                 # 神经网络模型
│   ├── trainer.py               # 训练器
│   ├── app.py                   # Flask API服务
│   ├── requirements.txt         # Python依赖
│   └── Dockerfile               # Docker构建文件
├── docker compose.yml           # Docker编排文件
├── .gitignore                   # Git忽略文件
└── README.md                    # 项目说明
```

## 技术栈

- Python 3.11
- PyTorch 2.0+
- scikit-learn（TF-IDF特征提取）
- Flask（REST API）
- Docker（容器化部署）
