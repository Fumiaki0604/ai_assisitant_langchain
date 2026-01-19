@echo off
REM ECS Fargate デプロイスクリプト (Windows)

setlocal enabledelayedexpansion

set APP_NAME=slack-ai-assistant
set AWS_REGION=us-west-2

echo ============================================
echo Slack AI Assistant - ECS Fargate Deploy
echo ============================================

REM AWS Account ID取得
for /f "tokens=*" %%i in ('aws sts get-caller-identity --query Account --output text') do set AWS_ACCOUNT_ID=%%i
set ECR_REPO=%AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%APP_NAME%

echo AWS Account: %AWS_ACCOUNT_ID%
echo Region: %AWS_REGION%
echo ECR Repo: %ECR_REPO%
echo.

REM Step 1: ECRリポジトリ作成
echo [1/5] ECRリポジトリを作成...
aws ecr describe-repositories --repository-names %APP_NAME% --region %AWS_REGION% 2>nul || aws ecr create-repository --repository-name %APP_NAME% --region %AWS_REGION%

REM Step 2: ECRログイン
echo [2/5] ECRにログイン...
aws ecr get-login-password --region %AWS_REGION% | docker login --username AWS --password-stdin %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com

REM Step 3: Dockerイメージビルド＆プッシュ
echo [3/5] Dockerイメージをビルド...
cd ..
docker build -t %APP_NAME%:latest .

echo ECRにプッシュ...
docker tag %APP_NAME%:latest %ECR_REPO%:latest
docker push %ECR_REPO%:latest
cd deploy

REM Step 4: Secrets Manager確認
echo [4/5] Secrets Manager設定を確認...
echo 以下のシークレットが必要です:
echo   - %APP_NAME%/pinecone-api-key
echo   - %APP_NAME%/slack-bot-token
echo   - %APP_NAME%/slack-app-token
echo   - %APP_NAME%/slack-signing-secret
echo.

REM Step 5: CloudFormationデプロイ
echo [5/5] CloudFormationスタックをデプロイ...
aws cloudformation deploy --template-file cloudformation.yml --stack-name %APP_NAME% --parameter-overrides AppName=%APP_NAME% ECRImageUri=%ECR_REPO%:latest --capabilities CAPABILITY_NAMED_IAM --region %AWS_REGION%

echo.
echo ============================================
echo デプロイ完了!
echo ============================================
echo.
echo ログ確認:
echo   aws logs tail /ecs/%APP_NAME% --follow --region %AWS_REGION%
echo.

endlocal
