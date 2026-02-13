# Slack AI Assistant

RAG（検索拡張生成）を使用してSlack内の質問に自動回答するPythonアプリケーション。
社内ナレッジに該当がない場合もWeb検索と一般知識で「判断の軸」を整理して返す、**逃げないAIアシスタント**。

## 技術スタック

- **言語**: Python 3.11+
- **LLM**: AWS Bedrock (Claude 3.5 Sonnet)
- **ベクトルDB**: Pinecone (Amazon Titan Embeddings V2, 1024次元)
- **リランキング**: Cohere Rerank Multilingual v3
- **Slackフレームワーク**: slack-bolt (Socket Mode)
- **インフラ**: AWS ECS Fargate / ECR / CloudWatch / Secrets Manager

## 処理フロー

```
Slack Message
  ↓
メッセージ意図分類（LLM）
  ├─ 共有・連絡 → 自然なリアクションを返して終了
  └─ 質問・相談 ↓
Pinecone検索（similarity_score_threshold=0.5）
  ↓
Cohere Rerank（min_score=0.5）
  ↓
内部ナレッジ十分？
  ├─ Yes → RAGプロンプトで回答生成 → Grounding検証
  └─ No  → Web検索フォールバック（コンテンツ取得）
           → 3層構造プロンプトで回答生成
             ① 一般論・業界の共通認識
             ② 当社視点での見解・仮説
             ③ 判断軸の整理
  ↓
Slack Reply（参考情報 + 信頼度スコア + フィードバックボタン）
```

## 機能

- **自動返信チャンネル**: 指定チャンネルの全メッセージに自動応答
- **メンション応答**: `@bot` で任意のチャンネルから質問可能
- **メッセージ意図分類**: 質問/共有を自動判定し、共有にはリアクションのみ返す
- **2段階フィルタリング**: Pinecone類似度閾値 + Cohere Rerankで無関係ドキュメントを除外
- **Web検索フォールバック**: 社内ナレッジ不足時にDuckDuckGoで検索しページ内容も取得
- **質問タイプ分類**: 自社製品 vs 外部サービスの質問を判定しWeb検索の要否を決定
- **回答のGrounding検証**: 回答が参考資料に基づいているかLLMで検証
- **信頼度スコア**: 関連度 x 整合性で信頼度を表示
- **フィードバック**: 👍/👎 ボタン・リアクションで回答品質を記録
- **人間優先**: 人間が先に返信済みのスレッドはスキップ
- **複数データソース**: Slack履歴 / Google Drive / Notion / PDF / Markdown

## セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

`.env.example`をコピーして`.env`を作成:

```bash
cp .env.example .env
```

必須環境変数:
| 変数 | 説明 |
|---|---|
| `PINECONE_API_KEY` | Pinecone API キー |
| `PINECONE_ENVIRONMENT` | Pinecone 環境 |
| `SLACK_BOT_TOKEN` | Slack Bot OAuth トークン |
| `SLACK_APP_TOKEN` | Slack App レベルトークン |
| `SLACK_SIGNING_SECRET` | Slack署名シークレット |
| `SLACK_AUTO_REPLY_CHANNELS` | 自動返信対象チャンネルID（カンマ区切り） |
| `COHERE_API_KEY` | Cohere API キー（リランキング用） |

AWS認証はAWS CLI設定済み or IAMロール（ECS）。

### 3. Pineconeインデックス作成

```bash
python scripts/setup_pinecone.py
```

### 4. ドキュメント登録

```bash
python scripts/load_all_documents.py --slack CME3BV4PN    # Slack履歴
python scripts/load_all_documents.py --files ./documents  # ファイル
python scripts/load_all_documents.py --all                # 全ソース
```

### 5. 起動

```bash
python src/slack/bot.py
```

Docker:
```bash
docker-compose up -d --build
```

## ディレクトリ構造

```
slack-ai-assistant/
├── src/
│   ├── slack/bot.py           # メインエントリポイント（イベント処理）
│   ├── rag/rag_service.py     # RAGサービス（検索・リランク・回答生成）
│   ├── rag/embeddings.py      # Amazon Titan Embeddings V2
│   ├── llm/bedrock.py         # Bedrock Claude 3.5 Sonnet
│   ├── loaders/               # データローダー（Slack/ファイル/Notion/Google Drive）
│   ├── feedback/              # フィードバック記録
│   ├── evaluation/            # RAGAS評価フレームワーク
│   └── auth/                  # Google認証
├── config/settings.py         # Pydantic Settings（.envから自動読み込み）
├── deploy/cloudformation.yml  # AWS CloudFormationテンプレート
├── scripts/                   # ユーティリティスクリプト
└── data/                      # ローカルデータ（gitignore対象）
```

## AWS デプロイ

```bash
# ビルド・プッシュ
docker build -t slack-ai-assistant:latest .
docker tag slack-ai-assistant:latest 433864970174.dkr.ecr.us-west-2.amazonaws.com/slack-ai-assistant:latest
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 433864970174.dkr.ecr.us-west-2.amazonaws.com
docker push 433864970174.dkr.ecr.us-west-2.amazonaws.com/slack-ai-assistant:latest

# デプロイ
aws ecs update-service --cluster slack-ai-assistant-cluster --service slack-ai-assistant-service --force-new-deployment --region us-west-2

# ログ確認
aws logs tail /ecs/slack-ai-assistant --follow --region us-west-2
```

## 設計思想

- **逃げない回答**: 社内データがなくても「回答できません」で終わらせない。一般論 → 当社仮説 → 判断軸の3層で整理する
- **共有には共感**: 質問でないメッセージには無理に回答せず、自然なリアクションを返す
- **ノイズ除去**: 2段階フィルタリング（Pinecone閾値 + Cohere Rerank）で無関係ドキュメントを確実に弾く
- **透明性**: 参考情報のソース・信頼度スコアを必ず表示し、根拠を明示する
