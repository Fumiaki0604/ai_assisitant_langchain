"""
全ドキュメントソースからデータを読み込んでPineconeに登録
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.loaders.slack_loader import load_slack_history
from src.loaders.file_loader import load_documents_from_directory
from src.loaders.notion_loader import load_notion_pages
from config.settings import settings
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_all(slack_channels: list = None, documents_dir: str = None, include_notion: bool = False):
    """
    全ソースからドキュメントを読み込み
    """
    results = {}

    # Slack履歴
    if slack_channels:
        print("\n" + "="*50)
        print("Slack履歴を読み込み中...")
        print("="*50)

        for channel_id in slack_channels:
            print(f"\nチャンネル: {channel_id}")
            success, count = load_slack_history(channel_id, limit=100)
            results[f"slack_{channel_id}"] = {"success": success, "count": count}
            print(f"  -> {count} 件登録")

    # ファイルローダー
    if documents_dir and os.path.exists(documents_dir):
        print("\n" + "="*50)
        print(f"ファイルを読み込み中: {documents_dir}")
        print("="*50)

        success, count = load_documents_from_directory(documents_dir)
        results["files"] = {"success": success, "count": count}
        print(f"  -> {count} 件登録")

    # Notion
    if include_notion and settings.notion_api_token:
        print("\n" + "="*50)
        print("Notionページを読み込み中...")
        print("="*50)

        success, count = load_notion_pages(limit=50)
        results["notion"] = {"success": success, "count": count}
        print(f"  -> {count} 件登録")

    # 結果サマリー
    print("\n" + "="*50)
    print("読み込み結果サマリー")
    print("="*50)

    total = 0
    for source, result in results.items():
        status = "OK" if result["success"] else "SKIP"
        print(f"  {source}: {result['count']} 件 [{status}]")
        total += result["count"]

    print(f"\n合計: {total} 件のドキュメントを登録しました")
    print("="*50 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ドキュメントをPineconeに登録')
    parser.add_argument('--slack', nargs='*', help='Slackチャンネル ID (例: CME3BV4PN)')
    parser.add_argument('--files', type=str, help='ドキュメントディレクトリパス')
    parser.add_argument('--notion', action='store_true', help='Notionページを読み込む')
    parser.add_argument('--all', action='store_true', help='全ソースから読み込む')

    args = parser.parse_args()

    # デフォルト値
    default_slack_channels = ["CME3BV4PN"]
    default_documents_dir = os.path.join(os.path.dirname(__file__), '../documents')

    if args.all:
        # 全ソースから読み込み
        load_all(
            slack_channels=default_slack_channels,
            documents_dir=default_documents_dir,
            include_notion=True
        )
    else:
        # 指定されたソースのみ
        slack_channels = args.slack if args.slack else None
        documents_dir = args.files if args.files else None
        include_notion = args.notion

        if not any([slack_channels, documents_dir, include_notion]):
            print("使用方法:")
            print("  python load_all_documents.py --all                    # 全ソースから読み込み")
            print("  python load_all_documents.py --slack CME3BV4PN        # Slack履歴のみ")
            print("  python load_all_documents.py --files ./documents      # ファイルのみ")
            print("  python load_all_documents.py --notion                 # Notionのみ")
            print("  python load_all_documents.py --slack CME3BV4PN --files ./documents --notion")
        else:
            load_all(
                slack_channels=slack_channels,
                documents_dir=documents_dir,
                include_notion=include_notion
            )
