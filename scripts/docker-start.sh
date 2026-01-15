#!/bin/bash
# Docker起動スクリプト

set -e

echo "==================================="
echo "Slack AI Assistant - Docker起動"
echo "==================================="

# .envファイルの確認
if [ ! -f .env ]; then
    echo "Error: .env ファイルが見つかりません"
    exit 1
fi

# ビルドと起動
echo "Dockerイメージをビルド中..."
docker-compose build

echo "コンテナを起動中..."
docker-compose up -d

echo ""
echo "==================================="
echo "起動完了!"
echo "==================================="
echo ""
echo "ログ確認: docker-compose logs -f"
echo "停止: docker-compose down"
echo ""
