#!/bin/bash
# ECS Fargate デプロイスクリプト

set -e

# 設定
APP_NAME="slack-ai-assistant"
AWS_REGION="us-west-2"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${APP_NAME}"

echo "============================================"
echo "Slack AI Assistant - ECS Fargate Deploy"
echo "============================================"
echo "AWS Account: ${AWS_ACCOUNT_ID}"
echo "Region: ${AWS_REGION}"
echo "ECR Repo: ${ECR_REPO}"
echo ""

# Step 1: ECRリポジトリ作成
echo "[1/5] ECRリポジトリを作成..."
aws ecr describe-repositories --repository-names ${APP_NAME} --region ${AWS_REGION} 2>/dev/null || \
aws ecr create-repository --repository-name ${APP_NAME} --region ${AWS_REGION}

# Step 2: ECRログイン
echo "[2/5] ECRにログイン..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Step 3: Dockerイメージビルド＆プッシュ
echo "[3/5] Dockerイメージをビルド..."
docker build -t ${APP_NAME}:latest ..

echo "ECRにプッシュ..."
docker tag ${APP_NAME}:latest ${ECR_REPO}:latest
docker push ${ECR_REPO}:latest

# Step 4: Secrets Manager設定確認
echo "[4/5] Secrets Manager設定を確認..."
echo "以下のシークレットが必要です:"
echo "  - ${APP_NAME}/pinecone-api-key"
echo "  - ${APP_NAME}/slack-bot-token"
echo "  - ${APP_NAME}/slack-app-token"
echo "  - ${APP_NAME}/slack-signing-secret"
echo ""
echo "まだ作成していない場合は、setup-secrets.sh を実行してください。"

# Step 5: CloudFormationデプロイ
echo "[5/5] CloudFormationスタックをデプロイ..."
aws cloudformation deploy \
  --template-file cloudformation.yml \
  --stack-name ${APP_NAME} \
  --parameter-overrides \
    AppName=${APP_NAME} \
    ECRImageUri=${ECR_REPO}:latest \
  --capabilities CAPABILITY_NAMED_IAM \
  --region ${AWS_REGION}

echo ""
echo "============================================"
echo "デプロイ完了!"
echo "============================================"
echo ""
echo "ログ確認:"
echo "  aws logs tail /ecs/${APP_NAME} --follow --region ${AWS_REGION}"
echo ""
echo "サービス状態確認:"
echo "  aws ecs describe-services --cluster ${APP_NAME}-cluster --services ${APP_NAME}-service --region ${AWS_REGION}"
echo ""
