FROM python:3.12-slim

WORKDIR /app

# システム依存パッケージをインストール
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 依存関係をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# ログディレクトリを作成
RUN mkdir -p logs

# 環境変数
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# ボットを起動
CMD ["python", "src/slack/bot.py"]
