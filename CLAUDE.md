# CLAUDE.md

This file provides guidance to Claude Code when working with this Slack AI Assistant project.

## Project Overview

Slack AI Assistantは、Slack内の質問に対してRAG（検索拡張生成）を使用して自動回答するPythonアプリケーションです。

## 技術スタック

- **Python 3.11+**
- **LangChain**: LLMオーケストレーションフレームワーク
- **AWS Bedrock**: Claude 3.5 Sonnetを使用
- **Pinecone**: ベクトルデータベース
- **slack-bolt**: Slackアプリフレームワーク

## プロジェクト構造

```
slack-ai-assistant/
├── src/
│   ├── slack/              # Slackイベント処理
│   │   ├── bot.py          # Slackボット初期化
│   │   └── event_handler.py # イベントハンドラー
│   ├── rag/                # RAG関連ロジック
│   │   ├── embeddings.py   # 埋め込み生成
│   │   ├── vectorstore.py  # Pinecone操作
│   │   └── retriever.py    # 検索ロジック
│   ├── llm/                # LLM連携
│   │   └── bedrock.py      # AWS Bedrock連携
│   ├── data_sources/       # データソース連携
│   │   ├── slack_loader.py     # Slack履歴取得
│   │   ├── confluence_loader.py # Confluence連携
│   │   ├── notion_loader.py    # Notion連携
│   │   └── file_loader.py      # ファイル読込
│   ├── indexing/           # インデックス構築
│   │   └── index_builder.py # インデックス構築スクリプト
│   └── feedback/           # フィードバック処理
│       └── feedback_handler.py # リアクション収集
├── config/
│   └── settings.py         # 設定管理（Pydantic Settings）
├── scripts/
│   └── setup_index.py      # 初回インデックス構築
└── tests/                  # テストコード
```

## 重要な実装パターン

### 1. 設定管理

`config/settings.py`でPydantic Settingsを使用して環境変数を管理しています。

```python
from config.settings import settings

# 設定の使用例
region = settings.aws_region
model_id = settings.bedrock_model_id
```

### 2. AWS Bedrock連携

`src/llm/bedrock.py`でBedrockとの連携を実装しています。

AWS認証は以下の順序で自動取得されます：
1. 環境変数 (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
2. AWS CLI設定 (~/.aws/credentials)
3. IAMロール

```python
from src.llm.bedrock import get_bedrock_llm

llm = get_bedrock_llm()
response = llm.invoke("質問内容")
```

### 3. RAGパターン

LangChainのRAGパターンを使用します：

1. **ドキュメントのロード**: data_sources/内の各ローダー
2. **チャンク分割**: LangChainのTextSplitter
3. **埋め込み生成**: BedrockEmbeddings
4. **ベクトル保存**: Pinecone
5. **検索と生成**: Retriever + LLM

### 4. Slackイベント処理

slack-boltを使用してイベントを処理します：

```python
from slack_bolt import App

app = App(token=settings.slack_bot_token)

@app.event("app_mention")
def handle_mention(event, say):
    # メンション処理
    pass
```

## 開発ワークフロー

### 初回セットアップ

1. 仮想環境作成
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. 依存関係インストール
```bash
pip install -r requirements.txt
```

3. 環境変数設定
```bash
cp .env.example .env
# .envを編集して必要な情報を入力
```

4. AWS設定確認
```bash
aws configure  # AWS CLIで認証情報を設定済み
```

5. Bedrock接続テスト
```bash
python src/llm/bedrock.py
```

### データインデックス構築

```bash
python scripts/setup_index.py
```

### アプリケーション起動

```bash
python src/main.py
```

## トラブルシューティング

### AWS Bedrock接続エラー

- AWS CLIの設定を確認: `aws configure list`
- リージョンを確認: us-west-2でClaude 3.5 Sonnetが利用可能
- Bedrockモデルアクセスを確認: AWSコンソールでモデルアクセスを有効化

### Pinecone接続エラー

- API Keyを確認
- Index名が正しいか確認
- Environmentが正しいか確認

### Slack接続エラー

- Bot TokenとApp Tokenを確認
- Slack Appの権限を確認（chat:write, app_mentions:read等）
- Socket Modeが有効になっているか確認

## テスト

```bash
pytest tests/
```

## コーディング規約

- **フォーマット**: Black
- **リント**: Flake8
- **型チェック**: mypy
- **ドキュメント**: Googleスタイルのdocstring

## セキュリティ

- `.env`ファイルは絶対にコミットしない
- AWS認証情報は環境変数またはAWS CLI設定で管理
- Slackトークンは環境変数で管理
- ログに機密情報を出力しない
