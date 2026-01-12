"""
Slackボットのメイン実装
"""
import sys
import os

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from config.settings import settings
from src.llm.bedrock import get_bedrock_llm
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Slackアプリの初期化
app = App(
    token=settings.slack_bot_token,
    signing_secret=settings.slack_signing_secret
)

# LLMインスタンスを取得
llm = get_bedrock_llm()

# 自動返信対象チャンネルのリスト
AUTO_REPLY_CHANNELS = [ch.strip() for ch in settings.slack_auto_reply_channels.split(',') if ch.strip()]


def has_human_reply(client, channel, thread_ts):
    """
    スレッド内に人間（ボット以外）の返信があるかチェック
    """
    try:
        # ボットのユーザーIDを取得
        bot_user_id = client.auth_test()["user_id"]

        # スレッドの返信を取得
        result = client.conversations_replies(
            channel=channel,
            ts=thread_ts,
            limit=100
        )

        messages = result.get("messages", [])

        # 最初のメッセージ（質問）以外をチェック
        for msg in messages[1:]:
            # ボット自身のメッセージはスキップ
            if msg.get("user") == bot_user_id or msg.get("bot_id"):
                continue

            # 人間の返信が見つかった
            logger.info(f"Human reply found in thread {thread_ts}")
            return True

        return False

    except Exception as e:
        logger.error(f"Error checking thread replies: {e}")
        return False


def has_bot_replied(client, channel, thread_ts):
    """
    ボットが既にスレッドに返信済みかチェック
    """
    try:
        # ボットのユーザーIDを取得
        bot_user_id = client.auth_test()["user_id"]

        # スレッドの返信を取得
        result = client.conversations_replies(
            channel=channel,
            ts=thread_ts,
            limit=100
        )

        messages = result.get("messages", [])

        # ボット自身の返信があるかチェック
        for msg in messages[1:]:  # 最初のメッセージ（質問）以外
            if msg.get("user") == bot_user_id or msg.get("bot_id"):
                logger.info(f"Bot has already replied to thread {thread_ts}")
                return True

        return False

    except Exception as e:
        logger.error(f"Error checking bot replies: {e}")
        return False


@app.event("app_mention")
def handle_mention(event, say, client):
    """
    ボットがメンションされた時の処理
    """
    try:
        # メンションを取得
        user = event["user"]
        text = event["text"]
        channel = event["channel"]
        thread_ts = event.get("thread_ts") or event["ts"]

        logger.info(f"Received mention from {user}: {text}")

        # ボットのメンションを削除してクリーンなテキストを取得
        clean_text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()

        # Claudeに質問を送信
        logger.info(f"Sending to Claude: {clean_text}")
        response = llm.invoke(clean_text)

        # Slackに返信（スレッドで返信）
        say(
            text=response.content,
            thread_ts=thread_ts
        )

        logger.info(f"Response sent to {user}")

    except Exception as e:
        logger.error(f"Error handling mention: {e}", exc_info=True)
        say(
            text=f"申し訳ございません。エラーが発生しました: {str(e)}",
            thread_ts=thread_ts
        )


@app.event("message")
def handle_message_events(event, say, client):
    """
    メッセージイベントの処理
    自動返信チャンネルでは全メッセージに反応
    """
    try:
        # サブタイプのあるメッセージ（編集、削除など）は無視
        if event.get("subtype"):
            return

        # ボット自身のメッセージは無視
        if event.get("bot_id"):
            return

        channel = event.get("channel")
        text = event.get("text", "")
        user = event.get("user")
        ts = event.get("ts")
        thread_ts = event.get("thread_ts")

        # 自動返信対象チャンネルでない場合はスキップ
        if channel not in AUTO_REPLY_CHANNELS:
            return

        # スレッド内のメッセージの場合
        if thread_ts:
            # 人間の返信がある場合はスキップ
            if has_human_reply(client, channel, thread_ts):
                logger.info(f"Skipping: Human has already replied to thread {thread_ts}")
                return

            # ボットが既に返信済みの場合はスキップ
            if has_bot_replied(client, channel, thread_ts):
                logger.info(f"Skipping: Bot has already replied to thread {thread_ts}")
                return

        # 新しい質問（スレッドでない）またはまだ誰も返信していないスレッド
        logger.info(f"Auto-replying to message from {user} in channel {channel}: {text}")

        # Claudeに質問を送信
        response = llm.invoke(text)

        # Slackに返信（スレッドで返信）
        say(
            text=response.content,
            thread_ts=thread_ts or ts  # 新規メッセージの場合はスレッドを開始
        )

        logger.info(f"Auto-response sent")

    except Exception as e:
        logger.error(f"Error handling message: {e}", exc_info=True)


def start_bot():
    """
    ボットを起動
    """
    logger.info("Starting Slack bot...")
    logger.info(f"Bot token: {settings.slack_bot_token[:20]}...")
    logger.info(f"App token: {settings.slack_app_token[:20]}...")
    logger.info(f"Auto-reply channels: {AUTO_REPLY_CHANNELS}")

    # Socket Modeでボットを起動
    handler = SocketModeHandler(app, settings.slack_app_token)
    handler.start()


if __name__ == "__main__":
    start_bot()
