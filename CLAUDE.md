# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules
- 回答は最大7行（箇条書き換算）
- 不明な場合は推測せず「不明」と明示
- コードは変更差分（diff）または変更箇所のみ
- ログはエラー前後など要点のみ（最大50行）

## Style
- 日本語
- 丁寧だが簡潔

## Project Overview

Slack AI Assistant - RAG（検索拡張生成）を使用してSlack内の質問に自動回答するPythonアプリケーション。AWS ECS Fargateで24時間稼働。

## Commands

```bash
# ローカル起動
python src/slack/bot.py

# Docker起動（ローカル）
docker-compose up -d --build

# ドキュメント登録
python scripts/load_all_documents.py --slack CME3BV4PN    # Slack履歴
python scripts/load_all_documents.py --files ./documents  # ファイル
python scripts/load_all_documents.py --all                # 全ソース

# Pineconeインデックス操作
python scripts/setup_pinecone.py    # インデックス作成
python scripts/reset_pinecone.py    # インデックス削除

# フィードバック確認
python scripts/view_feedback.py

# RAGテスト
python scripts/test_rag_search.py
```

## AWS Deployment

```bash
# ECRログイン
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 433864970174.dkr.ecr.us-west-2.amazonaws.com

# イメージビルド・プッシュ
docker build -t slack-ai-assistant:latest .
docker tag slack-ai-assistant:latest 433864970174.dkr.ecr.us-west-2.amazonaws.com/slack-ai-assistant:latest
docker push 433864970174.dkr.ecr.us-west-2.amazonaws.com/slack-ai-assistant:latest

# ECSサービス更新（新イメージ反映）
aws ecs update-service --cluster slack-ai-assistant-cluster --service slack-ai-assistant-service --force-new-deployment --region us-west-2

# ログ確認
aws logs tail /ecs/slack-ai-assistant --follow --region us-west-2

# CloudFormation再デプロイ
aws cloudformation deploy --template-file deploy/cloudformation.yml --stack-name slack-ai-assistant --parameter-overrides AppName=slack-ai-assistant ECRImageUri=433864970174.dkr.ecr.us-west-2.amazonaws.com/slack-ai-assistant:latest --capabilities CAPABILITY_NAMED_IAM --region us-west-2
```

## Architecture

### Core Flow
```
Slack Message → bot.py → RAGService → Pinecone (検索) → Bedrock Claude (生成) → Slack Reply
```

### Key Components

**src/slack/bot.py** - メインエントリポイント
- Socket Mode でSlackイベントを受信
- `@app.event("message")`: 自動返信（SLACK_AUTO_REPLY_CHANNELS内のみ）
- `@app.event("app_mention")`: メンション応答
- `@app.event("reaction_added")`: 👍/👎フィードバック記録
- 人間が先に返信済みのスレッドはスキップ

**src/rag/rag_service.py** - RAGサービス（シングルトン）
- `get_rag_service()` で取得
- `answer_question(question)` → `{"answer": str, "sources": list}`
- LangChain RetrievalQA + Pinecone + Bedrock

**src/rag/embeddings.py** - Amazon Titan Text Embeddings V2 (1024次元)

**src/llm/bedrock.py** - Claude 3.5 Sonnet (`anthropic.claude-3-5-sonnet-20241022-v2:0`)

**src/loaders/** - ドキュメントローダー
- `slack_loader.py`: Slack履歴（スレッド単位）
- `file_loader.py`: PDF/Word/Markdown/テキスト
- `notion_loader.py`: Notionページ

**config/settings.py** - Pydantic Settings（`.env`から自動読み込み）

### AWS Infrastructure (deploy/cloudformation.yml)
- **ECS Fargate**: slack-ai-assistant-cluster / slack-ai-assistant-service
- **ECR**: 433864970174.dkr.ecr.us-west-2.amazonaws.com/slack-ai-assistant
- **Secrets Manager**: 認証情報（PINECONE_API_KEY, SLACK_*トークン）
- **CloudWatch Logs**: /ecs/slack-ai-assistant
- **VPC**: 専用VPC + パブリックサブネット2つ

## Configuration

必須環境変数（.env / Secrets Manager）:
- `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`
- `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `SLACK_SIGNING_SECRET`
- `SLACK_AUTO_REPLY_CHANNELS`: 自動返信対象チャンネルID（カンマ区切り）
- AWS認証: AWS CLI設定済み or IAMロール（ECS）

## Slack App Requirements

必要なOAuthスコープ: `chat:write`, `app_mentions:read`, `channels:history`, `reactions:read`

Socket Mode: 有効化必須
