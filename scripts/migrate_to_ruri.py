"""
Pinecone インデックスを Titan(1024d) → Ruri v3-310m(768d) へ移行する

実行手順:
  1. python scripts/migrate_to_ruri.py
  2. ECS を再デプロイ（新しいイメージを反映）
     cd deploy && bash deploy.sh
     aws ecs update-service --cluster slack-ai-assistant-cluster \
       --service slack-ai-assistant-service --force-new-deployment --region us-west-2
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from pinecone import Pinecone, ServerlessSpec
from config.settings import settings


# ============================================================
# Step 1: Pinecone インデックスを再作成
# ============================================================

def recreate_pinecone_index():
    """既存インデックスを削除し、768次元（Ruri）で再作成する"""
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index_name = settings.pinecone_index_name
    existing = [idx['name'] for idx in pc.list_indexes()]

    if index_name in existing:
        logger.info(f"既存インデックスを削除: {index_name}")
        pc.delete_index(index_name)
        for _ in range(30):
            if index_name not in [idx['name'] for idx in pc.list_indexes()]:
                break
            logger.info("削除完了を待機中...")
            time.sleep(5)
        logger.info("削除完了")

    logger.info(f"新規インデックスを作成: {index_name} (dim=768, metric=dotproduct)")
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    for _ in range(30):
        status = pc.describe_index(index_name).status
        if status.get("ready", False):
            break
        logger.info("インデックス起動を待機中...")
        time.sleep(5)
    logger.info("インデックス準備完了")


# ============================================================
# Step 2: 各ソースからドキュメントを再インジェスト
# ============================================================

def reindex_google_drive():
    from src.loaders.google_drive_loader import GoogleDriveLoader
    try:
        loader = GoogleDriveLoader()
        folder_id = getattr(settings, 'google_drive_folder_id', None)
        logger.info(f"Google Drive: フォルダ {folder_id or 'すべて'} を読み込み中...")
        documents = loader.load_folder(folder_id)
        if documents:
            loader.save_to_pinecone(documents)
            logger.info(f"Google Drive: {len(documents)} 件登録完了")
        else:
            logger.warning("Google Drive: ドキュメントが見つかりませんでした")
    except Exception as e:
        logger.error(f"Google Drive: エラー - {e}", exc_info=True)


def reindex_slack(channel_ids: list):
    from src.loaders.slack_loader import load_slack_history
    for ch in channel_ids:
        try:
            logger.info(f"Slack: チャンネル {ch} を読み込み中...")
            success, count = load_slack_history(ch, limit=500)
            logger.info(f"Slack {ch}: {count} 件 ({'OK' if success else 'FAIL'})")
        except Exception as e:
            logger.error(f"Slack {ch}: エラー - {e}", exc_info=True)


def reindex_files():
    from src.loaders.file_loader import load_documents_from_directory
    documents_dir = os.path.join(os.path.dirname(__file__), '../documents')
    if not os.path.exists(documents_dir):
        logger.info(f"Files: {documents_dir} が存在しないためスキップ")
        return
    try:
        logger.info(f"Files: {documents_dir} を読み込み中...")
        success, count = load_documents_from_directory(documents_dir)
        logger.info(f"Files: {count} 件 ({'OK' if success else 'FAIL'})")
    except Exception as e:
        logger.error(f"Files: エラー - {e}", exc_info=True)


def reindex_notion():
    if not getattr(settings, 'notion_api_key', None):
        logger.info("Notion: API キーが未設定のためスキップ")
        return
    from src.loaders.notion_loader import load_notion_pages
    try:
        logger.info("Notion: ページを読み込み中...")
        success, count = load_notion_pages(limit=200)
        logger.info(f"Notion: {count} 件 ({'OK' if success else 'FAIL'})")
    except Exception as e:
        logger.error(f"Notion: エラー - {e}", exc_info=True)


# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Ruri v3-310m 移行スクリプト")
    print("=" * 60)
    print(f"対象インデックス: {settings.pinecone_index_name}")
    print()
    print("警告: 既存の Pinecone インデックスを削除・再作成します。")
    print("      実行中は Slack ボットの検索が一時的に機能しません。")
    print()
    confirm = input("実行しますか？ (yes/no): ")
    if confirm.strip().lower() != "yes":
        print("中止しました")
        sys.exit(0)

    # Slack チャンネル ID（再インジェスト対象、必要に応じて追加）
    SLACK_CHANNELS = ["CME3BV4PN"]

    print()
    logger.info("Step 1: Pinecone インデックスを再作成 (768次元)")
    recreate_pinecone_index()

    logger.info("Step 2: Google Drive ドキュメントを再インジェスト")
    reindex_google_drive()

    logger.info("Step 3: Slack 履歴を再インジェスト")
    reindex_slack(SLACK_CHANNELS)

    logger.info("Step 4: ローカルファイルを再インジェスト")
    reindex_files()

    logger.info("Step 5: Notion ページを再インジェスト")
    reindex_notion()

    print()
    print("=" * 60)
    print("移行完了!")
    print()
    print("次のステップ: ECS を再デプロイしてください")
    print("  cd deploy && bash deploy.sh")
    print("  aws ecs update-service \\")
    print("    --cluster slack-ai-assistant-cluster \\")
    print("    --service slack-ai-assistant-service \\")
    print("    --force-new-deployment --region us-west-2")
    print("=" * 60)
