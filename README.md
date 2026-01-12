# Slack AI Assistant

Slack内の質問スレッドに対して、過去の回答や社内ドキュメント・規約を参照してAIが自動回答するアプリケーション。

## 技術スタック

- **言語**: Python 3.11+
- **LLMフレームワーク**: LangChain
- **LLMプロバイダ**: AWS Bedrock (Claude 3.5 Sonnet)
- **ベクトルDB**: Pinecone
- **Slackフレームワーク**: slack-bolt

## 機能

- ✅ Slackでメンション時に自動回答
- ✅ RAG（検索拡張生成）による精度の高い回答
- ✅ 複数のデータソース対応
  - Slackの過去スレッド
  - 社内ドキュメント（Confluence/Notion）
  - 規約・マニュアル（PDF/Markdown/URL）
- ✅ フィードバック機能（👍/👎リアクション）

## セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

`.env.example`をコピーして`.env`を作成し、必要な情報を入力してください。

```bash
cp .env.example .env
```

### 3. データのインデックス構築

```bash
python scripts/setup_index.py
```

### 4. アプリケーションの起動

```bash
python src/main.py
```

## ディレクトリ構造

```
slack-ai-assistant/
├── src/
│   ├── slack/          # Slackイベント処理
│   ├── rag/            # RAG関連ロジック
│   ├── llm/            # LLM連携
│   ├── data_sources/   # データソース連携
│   ├── indexing/       # インデックス構築
│   └── feedback/       # フィードバック処理
├── config/             # 設定管理
├── scripts/            # ユーティリティスクリプト
├── tests/              # テストコード
└── data/               # ローカルデータ（gitignore対象）
```

## 開発者向け情報

詳細な実装ガイドは[CLAUDE.md](CLAUDE.md)を参照してください。
