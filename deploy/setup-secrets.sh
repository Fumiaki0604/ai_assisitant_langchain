#!/bin/bash
# Secrets Manager シークレット作成スクリプト

set -e

APP_NAME="slack-ai-assistant"
AWS_REGION="us-west-2"

echo "============================================"
echo "Secrets Manager セットアップ"
echo "============================================"
echo ""

# .envファイルから値を読み込み
if [ -f "../.env" ]; then
    source ../.env
else
    echo "Error: ../.env ファイルが見つかりません"
    exit 1
fi

# Pinecone API Key
echo "Pinecone API Key を登録..."
aws secretsmanager create-secret \
    --name "${APP_NAME}/pinecone-api-key" \
    --secret-string "${PINECONE_API_KEY}" \
    --region ${AWS_REGION} 2>/dev/null || \
aws secretsmanager update-secret \
    --secret-id "${APP_NAME}/pinecone-api-key" \
    --secret-string "${PINECONE_API_KEY}" \
    --region ${AWS_REGION}

# Slack Bot Token
echo "Slack Bot Token を登録..."
aws secretsmanager create-secret \
    --name "${APP_NAME}/slack-bot-token" \
    --secret-string "${SLACK_BOT_TOKEN}" \
    --region ${AWS_REGION} 2>/dev/null || \
aws secretsmanager update-secret \
    --secret-id "${APP_NAME}/slack-bot-token" \
    --secret-string "${SLACK_BOT_TOKEN}" \
    --region ${AWS_REGION}

# Slack App Token
echo "Slack App Token を登録..."
aws secretsmanager create-secret \
    --name "${APP_NAME}/slack-app-token" \
    --secret-string "${SLACK_APP_TOKEN}" \
    --region ${AWS_REGION} 2>/dev/null || \
aws secretsmanager update-secret \
    --secret-id "${APP_NAME}/slack-app-token" \
    --secret-string "${SLACK_APP_TOKEN}" \
    --region ${AWS_REGION}

# Slack Signing Secret
echo "Slack Signing Secret を登録..."
aws secretsmanager create-secret \
    --name "${APP_NAME}/slack-signing-secret" \
    --secret-string "${SLACK_SIGNING_SECRET}" \
    --region ${AWS_REGION} 2>/dev/null || \
aws secretsmanager update-secret \
    --secret-id "${APP_NAME}/slack-signing-secret" \
    --secret-string "${SLACK_SIGNING_SECRET}" \
    --region ${AWS_REGION}

echo ""
echo "============================================"
echo "シークレット登録完了!"
echo "============================================"
echo ""
echo "登録されたシークレット:"
aws secretsmanager list-secrets --region ${AWS_REGION} --query "SecretList[?starts_with(Name, '${APP_NAME}')].Name" --output table
echo ""
