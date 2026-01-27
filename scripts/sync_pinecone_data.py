"""
Pineconeデータ同期スクリプト
Slack履歴とGoogleDriveのデータを定期的に更新
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.loaders.slack_loader import SlackHistoryLoader
from src.loaders.google_drive_loader import GoogleDriveLoader
from config.settings import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sync_slack_history():
    """Slack履歴を同期"""
    channels = [ch.strip() for ch in settings.slack_auto_reply_channels.split(',') if ch.strip()]
    if not channels:
        logger.info("No Slack channels configured")
        return 0

    loader = SlackHistoryLoader()
    total_docs = 0

    for channel_id in channels:
        try:
            documents = loader.load_channel_threads(channel_id, limit=200)
            if documents:
                loader.save_to_pinecone(documents)
                total_docs += len(documents)
                logger.info(f"Synced {len(documents)} documents from Slack channel {channel_id}")
        except Exception as e:
            logger.error(f"Failed to sync Slack channel {channel_id}: {e}")

    return total_docs


def sync_google_drive():
    """GoogleDriveを同期"""
    folder_id = getattr(settings, 'google_drive_folder_id', None)

    try:
        loader = GoogleDriveLoader()
        documents = loader.load_folder(folder_id)

        if documents:
            loader.save_to_pinecone(documents)
            logger.info(f"Synced {len(documents)} documents from Google Drive")
            return len(documents)
    except Exception as e:
        logger.error(f"Failed to sync Google Drive: {e}")

    return 0


def main():
    """メイン処理"""
    logger.info("Starting Pinecone data sync...")

    slack_count = sync_slack_history()
    drive_count = sync_google_drive()

    logger.info(f"Sync completed: Slack={slack_count}, GoogleDrive={drive_count}")


if __name__ == "__main__":
    main()
