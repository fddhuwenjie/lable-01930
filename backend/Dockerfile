# 恶意URL分类服务 Dockerfile
# 使用跨平台基础镜像，支持 ARM 和 X86 架构

# 构建阶段
FROM --platform=$BUILDPLATFORM python:3.11-slim AS builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖到指定目录
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# 运行阶段
FROM --platform=$TARGETPLATFORM python:3.11-slim AS runtime

WORKDIR /app

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 从构建阶段复制已安装的依赖
COPY --from=builder /install /usr/local

# 复制应用代码
COPY . .

# 创建输出目录
RUN mkdir -p output

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 暴露端口
EXPOSE 5000

# 启动命令：先训练模型，再启动服务
CMD ["sh", "-c", "python trainer.py && python app.py"]
