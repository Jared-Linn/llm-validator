FROM python:3.12-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目
COPY . .

# 创建数据目录
RUN mkdir -p data/uploads

# 端口
EXPOSE 8900

# 默认启动
CMD ["python3", "app.py"]
