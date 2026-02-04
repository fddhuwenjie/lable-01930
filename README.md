# 恶意URL分类系统

基于监督学习的恶意URL分类项目，使用PyTorch构建全连接神经网络，通过TF-IDF特征提取实现URL的二分类（正常/恶意）。

## How to Run

### Docker方式（推荐）

```bash
# 构建并启动服务
docker-compose up --build -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
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

### 任务1：基于监督学习的恶意URL分类

#### 数据预处理
- 加载恶意URL数据集（包含URL文本和标签，标签为正常=0/恶意=1）
- 参考SpamDataLoader类实现URLDataLoader，完成数据读取
- 采用TF-IDF提取URL文本特征（最大特征数1200，过滤英文停用词）
- 将特征与标签转换为PyTorch张量
- 划分训练集（70%）、验证集（20%）、测试集（10%）

#### 模型构建
- 基于PyTorch继承nn.Module类
- 构建含1个隐藏层（128个神经元）的全连接神经网络
- 隐藏层激活函数使用ReLU
- 输出层适配二分类（无需额外激活，配合交叉熵损失）

#### 模型训练与评估
- 配置交叉熵损失函数（nn.CrossEntropyLoss）
- 学习率：5e-5
- 训练轮数：25轮
- 批次大小：32
- 每轮训练后计算验证集准确率
- 训练结束后计算测试集准确率
- 绘制训练/验证损失曲线
- 分析模型对恶意URL的识别效果

---

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
├── docker-compose.yml           # Docker编排文件
├── .gitignore                   # Git忽略文件
└── README.md                    # 项目说明
```

## 技术栈

- Python 3.11
- PyTorch 2.0+
- scikit-learn（TF-IDF特征提取）
- Flask（REST API）
- Docker（容器化部署）
