# PDF OCR处理工具 - Docker镜像
# 使用NVIDIA CUDA基础镜像支持GPU加速

FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置Python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 创建工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .
COPY requirements-api.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-api.txt

# 复制应用代码
COPY *.py ./
COPY config.template.yaml ./config.yaml

# 创建必要的目录
RUN mkdir -p /data/input /data/output /data/reference /data/temp

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "api_server.py"]
