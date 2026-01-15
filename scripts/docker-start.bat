@echo off
REM Docker起動スクリプト (Windows)

echo ===================================
echo Slack AI Assistant - Docker起動
echo ===================================

REM .envファイルの確認
if not exist .env (
    echo Error: .env ファイルが見つかりません
    exit /b 1
)

REM ビルドと起動
echo Dockerイメージをビルド中...
docker-compose build

echo コンテナを起動中...
docker-compose up -d

echo.
echo ===================================
echo 起動完了!
echo ===================================
echo.
echo ログ確認: docker-compose logs -f
echo 停止: docker-compose down
echo.
