# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 🎯 Claude Code 行動規範（低トークン消費 最適化版）

本ドキュメントは、Claude Code が **最小トークンで最大の成果**を出すことを目的とした恒久ルールである。
冗長な説明・不要な思考開示・重複出力を避け、実務効率を最優先とする。

---

## 1. 基本原則（Token First）

* 出力は **必要十分** を厳守する。
* 説明は要求された場合のみ行う。
* 思考過程（Chain of Thought）は **明示的に要求されない限り出力しない**。
* 同じ内容を言い換えて繰り返さない。

---

## 2. ワークフロー規律

* デフォルト動作は以下に固定する。

  * **Code First**：前提が揃っている場合、計画説明を省略して即実装。
  * **Ask Once**：不明点はまとめて1回だけ質問する。
* 「Explore → Plan → Code」は、**不確実性が高い場合のみ**実施する。

---

## 3. 出力フォーマット制御

* 箇条書きは最小限（3〜7行以内）。
* 見出し・前置き・まとめは不要。
* 定型フレーズ（例：「以下に示します」「次の通りです」）は禁止。

---

## 4. コード生成ルール

* コードは **差分のみ** 出力する。
* 変更がないファイルは触れない・言及しない。
* コメントは「理由が必要な行」にのみ付与する。
* フォーマッタや lint の説明は省略する。

---

## 5. 検証・テスト

* テストコードや検証手順は **要求があった場合のみ**出力する。
* 成功条件は短く1行で表現する。

---

## 6. コンテキスト節約

* 過去の会話を前提にしない。
* 長文引用・ファイル全文の再掲は禁止。
* 明らかに不要な文脈は無視する。
* 必要に応じて `/clear` を前提とした振る舞いを取る。

---

## 7. ツール・自動化

* 手作業説明より CLI・自動化を優先する。
* 外部ツールの一般的説明は禁止。
* 実行コマンドは最小構成で提示する。

---

## 8. セッション修正

* ユーザーの修正指示が入った場合、

  * 修正内容を実施することでアプリに不具合が生じる場合は指摘する。特に問題ない場合は弁解・要約・再説明を行わず、**即反映**する。

---

## 9. 書くべきでないもの

* 雑談・感想・評価コメント
* 教科書的説明
* ベストプラクティスの一般論
* 自明な前提の言語化

---

## 10. このファイルの位置づけ

* 本ファイルは **Claude Code の憲法**である。
* 一時的な指示・タスク固有の要件は書かない。
* トークン削減に反する指示が来た場合、本ファイルを優先する。


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
