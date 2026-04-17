# Slack AI Assistant

RAG（検索拡張生成）を使用してSlack内の質問に自動回答するPythonアプリケーション。
社内ナレッジに該当がない場合もWeb検索と一般知識で「判断の軸」を整理して返す、**逃げないAIアシスタント**。

## 技術スタック

- **言語**: Python 3.12
- **LLM**: AWS Bedrock (`us.anthropic.claude-sonnet-4-6`)
- **ベクトルDB**: Pinecone (Amazon Titan Embeddings V2, 1024次元) + **Hybrid Search** (dense + BM25 sparse)
- **リランキング**: Cohere Rerank Multilingual v3（オプション）
- **Slackフレームワーク**: slack-bolt (Socket Mode)
- **インフラ**: AWS ECS Fargate / ECR / CloudWatch / Secrets Manager

## 処理フロー

```
Slack Message
  ↓
メッセージ意図分類（LLM）
  ├─ 共有・連絡 → 自然なリアクションを返して終了
  └─ 質問・相談 ↓
Slackリクエストタイプ分類（LLM）
  ├─ KNOWLEDGE  → 通常RAGフロー
  ├─ EXPERIENCE → 社内実績探索（Web検索・説明禁止）
  ├─ DOCUMENT   → 社内資料探索（Web検索・説明禁止）
  ├─ OWNER      → 担当者探索（Web検索・説明禁止）
  └─ OPINION    → 通常RAGフロー
Pinecone Hybrid Search（dense Titan + sparse BM25, alpha=0.7）
  ↓
Cohere Rerank（min_score=0.5）または フォールバックスコアリング
  ↓
内部ナレッジ十分？（KNOWLEDGEのみ判定）
  ├─ Yes → RAGプロンプトで回答生成 → Grounding検証
  └─ No  → Web検索フォールバック（DuckDuckGo + コンテンツ取得）
           → 3層構造プロンプトで回答生成
  ↓
Slack Reply（参考情報 + 信頼度スコア + フィードバックボタン）
  ↓
自動採点（LLM / 5軸100点）→ S3 or ローカルに記録
```

## 機能

- **自動返信チャンネル**: 指定チャンネルの全メッセージに自動応答（メンション不要）
- **メンション応答**: `@bot` で任意のチャンネルから質問可能
- **チャンネル別Web検索制御**: `SLACK_NO_WEB_SEARCH_CHANNELS` でWeb検索を無効化（ヘルプデスク等）
- **メッセージ意図分類**: 質問/共有を自動判定し、共有にはリアクションのみ返す
- **5タイプリクエスト分類**: KNOWLEDGE / EXPERIENCE / DOCUMENT / OWNER / OPINION を自動判定
- **社内探索モード**: EXPERIENCE/DOCUMENT/OWNER はWeb検索・説明を省略し「あるかないか」だけ返す
- **Hybrid Search**: dense（意味検索）+ sparse（BM25キーワード）で固有名詞の精度を向上
- **2段階フィルタリング**: Pinecone Hybrid Search + Cohere Rerankで無関係ドキュメントを除外
- **Web検索フォールバック**: 社内ナレッジ不足時にDuckDuckGoで検索しページ内容も取得
- **回答のGrounding検証**: 回答が参考資料に基づいているかLLMで検証
- **信頼度スコア**: 関連度 × 整合性で信頼度を表示（🟢高 / 🟡中 / 🔴低）
- **フィードバック**: 👍/👎 ボタン・リアクションで回答品質を記録
- **自動採点ログ**: 回答ごとに5軸100点ルーブリックでLLM採点しS3/ローカルに記録
- **人間優先**: 人間が先に返信済みのスレッドはスキップ
- **複数データソース**: Slack履歴 / Google Drive / Notion / PDF / Markdown
- **Google Drive OCR**: スキャンPDF・画像PDFをOCRで正確にテキスト抽出
- **週次自動sync（差分更新）**: ECS Scheduled TaskでSlack履歴・Google DriveをPineconeへ増分同期（S3で状態管理）
- **画像対応**: Slackに添付された画像をBedrockのマルチモーダルで解析して回答

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
| `SLACK_APP_TOKEN` | Slack App レベルトークン（Socket Mode用） |
| `SLACK_SIGNING_SECRET` | Slack署名シークレット |
| `SLACK_AUTO_REPLY_CHANNELS` | 自動返信対象チャンネルID（カンマ区切り） |

任意環境変数:

| 変数 | 説明 |
|---|---|
| `SLACK_NO_WEB_SEARCH_CHANNELS` | Web検索を無効にするチャンネルID（カンマ区切り） |
| `SLACK_KNOWLEDGE_CHANNELS` | RAG取り込みのみ（返信なし）のチャンネルID |
| `COHERE_API_KEY` | Cohere API キー（リランキング精度向上） |
| `NOTION_API_KEY` | Notion インテグレーションシークレット |
| `GOOGLE_DRIVE_FOLDER_ID` | Google DriveフォルダID |
| `S3_STATE_BUCKET` | 増分同期の状態管理バケット名 |
| `PINECONE_HYBRID_ALPHA` | dense/sparse比率（デフォルト: 0.7） |

AWS認証はAWS CLI設定済み or IAMロール（ECS）。

### 3. Pineconeインデックス作成

```bash
python scripts/setup_pinecone.py
```

> インデックスは `dotproduct` メトリクスで作成されます（Hybrid Search必須）。

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
│   ├── slack/
│   │   ├── bot.py             # メインエントリポイント（イベント処理）
│   │   └── image_handler.py   # Slack添付画像の取得・リサイズ・base64変換
│   ├── rag/
│   │   ├── rag_service.py     # RAGサービス（検索・リランク・回答生成）
│   │   ├── classifier.py      # LLM分類器（意図・リクエストタイプ判定）
│   │   ├── reranker.py        # Cohere Rerank + フォールバック / Grounding検証
│   │   ├── web_searcher.py    # DuckDuckGo検索 / URLコンテンツ取得
│   │   ├── prompts.py         # プロンプトテンプレート
│   │   └── embeddings.py      # Amazon Titan Embeddings V2
│   ├── llm/bedrock.py         # Bedrock LLM
│   ├── loaders/               # データローダー（Slack / ファイル / Notion / Google Drive）
│   ├── loaders/sync_state.py  # 増分同期の状態管理（S3/ローカル）
│   ├── feedback/              # フィードバック記録
│   ├── evaluation/            # 5軸ルーブリック自動採点・ログ保存
│   └── auth/                  # Google認証
├── config/settings.py         # Pydantic Settings（.envから自動読み込み）
├── deploy/cloudformation.yml  # AWS CloudFormationテンプレート
├── scripts/
│   ├── sync_pinecone_data.py  # 増分同期スクリプト（ECS Scheduled Task）
│   ├── eval_report.py         # 採点ログ集計レポート
│   └── ...                    # その他ユーティリティ
└── data/                      # ローカルデータ（gitignore対象）
```

## AWS デプロイ

```bash
# ECRログイン
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 433864970174.dkr.ecr.us-west-2.amazonaws.com

# ビルド・プッシュ
docker build -t slack-ai-assistant:latest .
docker tag slack-ai-assistant:latest 433864970174.dkr.ecr.us-west-2.amazonaws.com/slack-ai-assistant:latest
docker push 433864970174.dkr.ecr.us-west-2.amazonaws.com/slack-ai-assistant:latest

# デプロイ
aws ecs update-service --cluster slack-ai-assistant-cluster --service slack-ai-assistant-service --force-new-deployment --region us-west-2

# ログ確認
aws logs tail /ecs/slack-ai-assistant --follow --region us-west-2
```

## メッセージ意図分類の挙動

ボットは受信メッセージを `question`（質問）/ `share`（共有・報告）に分類してから応答を決定する。

| 分類 | 投稿例 | ボットの動作 |
|---|---|---|
| `question` | 「Airecoの最新の汎用資料をお持ちの方がいらっしゃったら共有いただけると嬉しいです」 | RAGで検索し、社内ナレッジから該当資料を返答 |
| `question` | 「メルカートのオプション機能一覧はどこで見れますか？」 | RAGで検索して回答 |
| `share` | 「jQuery4系対応関連のドキュメントをざっと作成しました（初稿）」 | 共感・acknowledgmentのみ返す（RAGは動かない） |
| `share` | 「本日の定例MTGの議事録を共有します」 | 共感・acknowledgmentのみ返す |

## 回答品質の自動採点ログ

### 採点フレーム（100点 / 5軸 × 20点）

| 軸 | 見ているポイント |
|---|---|
| ①質問タイプ理解 | KNOWLEDGE/EXPERIENCE/DOCUMENT/OWNER/OPINION を正しく認識しているか |
| ②質問への直接回答 | 冒頭1文で「ある/ない/〜です」と直接答えているか |
| ③不要情報の少なさ | 余分な説明・トレンド解説・外部URLがないか |
| ④社内文脈理解 | Slackの社内会話トーンに合っているか |
| ⑤次の行動の妥当性 | 適切な次アクションを示しているか（EXPERIENCE/OWNERでは提案不要） |

```bash
python scripts/eval_report.py          # 過去30日
python scripts/eval_report.py --days 7 # 過去7日
```

## Google Drive OCRについての注意

PDFのテキスト抽出時に、Google DriveにOCR用の一時Googleドキュメントファイルが作成される。通常は処理後に自動削除されるが、削除に失敗した場合はログに警告が出るため、手動でGoogle Driveから削除すること。

```
Warning: Failed to delete temp OCR doc {id}: ... Please delete it manually from Google Drive.
```
