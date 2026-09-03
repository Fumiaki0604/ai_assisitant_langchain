FROM python:3.12-slim

WORKDIR /app

# システム依存パッケージをインストール
RUN apt-get update && apt-get install -y \
    gcc \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# 依存関係をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')" && \
    python -c "import requests, os; os.makedirs('/app/data', exist_ok=True); r = requests.get('https://storage.googleapis.com/pinecone-datasets-dev/bm25_params/msmarco_bm25_params_v4_0_0.json', timeout=60); r.raise_for_status(); open('/app/data/bm25_params.json', 'wb').write(r.content); print('BM25 params downloaded')" && \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('hotchpotch/japanese-reranker-small-v2', max_length=512); print('HF reranker model downloaded')"

# アプリケーションコードをコピー
COPY . .

# ログディレクトリを作成
RUN mkdir -p logs

# 環境変数
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# ボットを起動
CMD ["python", "src/slack/bot.py"]
