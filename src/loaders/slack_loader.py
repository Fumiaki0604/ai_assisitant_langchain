"""
Slack履歴をPineconeに登録するローダー
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from src.loaders.pinecone_storage import save_to_pinecone as _save_to_pinecone
from config.settings import settings
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SlackHistoryLoader:
    """
    Slack履歴をPineconeに登録するクラス
    """

    def __init__(self):
        self.client = WebClient(token=settings.slack_bot_token)

    def get_channel_name(self, channel_id: str) -> str:
        """チャンネル名を取得"""
        try:
            result = self.client.conversations_info(channel=channel_id)
            return result["channel"]["name"]
        except SlackApiError as e:
            logger.error(f"Failed to get channel name: {e}")
            return channel_id

    def get_permalink(self, channel_id: str, message_ts: str) -> str:
        """メッセージのパーマリンクを取得"""
        try:
            result = self.client.chat_getPermalink(channel=channel_id, message_ts=message_ts)
            return result["permalink"]
        except SlackApiError as e:
            logger.warning(f"Failed to get permalink for {channel_id}/{message_ts}: {e}")
            return ""

    def get_user_name(self, user_id: str) -> str:
        """ユーザー名を取得"""
        try:
            result = self.client.users_info(user=user_id)
            user = result["user"]
            return user.get("real_name") or user.get("display_name") or user.get("name") or user_id
        except SlackApiError as e:
            logger.error(f"Failed to get user name: {e}")
            return user_id

    def fetch_channel_history(self, channel_id: str, limit: int = 100) -> list:
        """
        チャンネルの履歴を取得
        """
        messages = []
        try:
            result = self.client.conversations_history(
                channel=channel_id,
                limit=limit
            )
            messages = result.get("messages", [])
            logger.info(f"Fetched {len(messages)} messages from channel {channel_id}")
        except SlackApiError as e:
            logger.error(f"Failed to fetch channel history: {e}")
        return messages

    def fetch_thread_replies(self, channel_id: str, thread_ts: str) -> list:
        """
        スレッドの返信を取得
        """
        try:
            result = self.client.conversations_replies(
                channel=channel_id,
                ts=thread_ts,
                limit=100
            )
            return result.get("messages", [])
        except SlackApiError as e:
            logger.error(f"Failed to fetch thread replies: {e}")
            return []

    def format_thread_as_document(self, channel_id: str, channel_name: str, messages: list) -> dict:
        """
        スレッドをドキュメント形式に整形
        """
        if not messages:
            return None

        # 最初のメッセージ（質問）
        first_msg = messages[0]
        question_user = self.get_user_name(first_msg.get("user", "Unknown"))
        question_text = first_msg.get("text", "")
        timestamp = first_msg.get("ts", "")

        # タイムスタンプを日付に変換
        try:
            dt = datetime.fromtimestamp(float(timestamp))
            date_str = dt.strftime("%Y-%m-%d %H:%M")
        except:
            date_str = "Unknown"

        # スレッドの内容を整形
        content_parts = [f"質問: {question_text}"]

        # 返信を追加
        for msg in messages[1:]:
            user = self.get_user_name(msg.get("user", "Unknown"))
            text = msg.get("text", "")
            if text:
                content_parts.append(f"回答 ({user}): {text}")

        content = "\n\n".join(content_parts)

        permalink = self.get_permalink(channel_id, timestamp)

        return {
            "content": content,
            "metadata": {
                "source": "slack",
                "channel_id": channel_id,
                "channel_name": channel_name,
                "thread_ts": timestamp,
                "date": date_str,
                "permalink": permalink,
                "title": f"Slack: {question_text[:50]}..." if len(question_text) > 50 else f"Slack: {question_text}"
            }
        }

    def load_channel_threads(self, channel_id: str, limit: int = 100) -> list:
        """
        チャンネルのスレッドをドキュメントとして読み込み
        """
        channel_name = self.get_channel_name(channel_id)
        messages = self.fetch_channel_history(channel_id, limit)

        documents = []
        thread_count = 0

        for msg in messages:
            # スレッドの親メッセージのみを対象
            thread_ts = msg.get("thread_ts")
            if thread_ts and thread_ts == msg.get("ts"):
                # スレッドの返信を取得
                thread_messages = self.fetch_thread_replies(channel_id, thread_ts)

                if len(thread_messages) > 1:  # 返信がある場合のみ
                    doc = self.format_thread_as_document(channel_id, channel_name, thread_messages)
                    if doc:
                        documents.append(doc)
                        thread_count += 1

            # スレッドでない単独メッセージも取り込み（オプション）
            elif not msg.get("thread_ts"):
                text = msg.get("text", "")
                if text and len(text) > 20:  # 短すぎるメッセージは除外
                    user = self.get_user_name(msg.get("user", "Unknown"))
                    timestamp = msg.get("ts", "")
                    try:
                        dt = datetime.fromtimestamp(float(timestamp))
                        date_str = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        date_str = "Unknown"

                    permalink = self.get_permalink(channel_id, timestamp)
                    documents.append({
                        "content": f"投稿 ({user}): {text}",
                        "metadata": {
                            "source": "slack",
                            "channel_id": channel_id,
                            "channel_name": channel_name,
                            "thread_ts": timestamp,
                            "date": date_str,
                            "permalink": permalink,
                            "title": f"Slack: {text[:50]}..." if len(text) > 50 else f"Slack: {text}"
                        }
                    })

        logger.info(f"Loaded {len(documents)} documents ({thread_count} threads) from #{channel_name}")
        return documents

    def save_to_pinecone(self, documents: list) -> bool:
        """
        ドキュメントをPineconeに保存
        """
        return _save_to_pinecone(documents)


def load_slack_history(channel_id: str, limit: int = 100):
    """
    Slack履歴を読み込んでPineconeに保存
    """
    loader = SlackHistoryLoader()
    documents = loader.load_channel_threads(channel_id, limit)

    if documents:
        success = loader.save_to_pinecone(documents)
        return success, len(documents)
    return False, 0


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # デフォルトのチャンネルを読み込み
    channel_id = "CME3BV4PN"
    print(f"\nSlack履歴を読み込み中: {channel_id}")

    success, count = load_slack_history(channel_id, limit=50)

    if success:
        print(f"\n完了: {count} 件のドキュメントをPineconeに登録しました")
    else:
        print("\n登録するドキュメントがありませんでした")
