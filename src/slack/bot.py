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
from src.rag.rag_service import get_rag_service
from src.feedback.feedback_logger import get_feedback_logger
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Slackアプリの初期化
app = App(
    token=settings.slack_bot_token,
    signing_secret=settings.slack_signing_secret
)

# RAGサービスを取得
rag_service = get_rag_service()

# フィードバックロガーを取得
feedback_logger = get_feedback_logger()

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

        # RAGで回答を生成
        logger.info(f"Processing question with RAG: {clean_text}")
        result = rag_service.answer_question(clean_text)

        # 回答を整形
        answer_text = result['answer']

        # 参考ドキュメントがある場合は追記
        if result['sources']:
            answer_text += "\n\n_参考ドキュメント:_"
            for i, source in enumerate(result['sources'][:2], 1):  # 最大2件
                answer_text += f"\n• {source['title']}"

        # Block Kitでフィードバックボタン付きメッセージを送信
        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": answer_text}
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "_この回答は役に立ちましたか？_"}
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "👍 良い", "emoji": True},
                        "style": "primary",
                        "action_id": "feedback_positive",
                        "value": clean_text
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "👎 改善が必要", "emoji": True},
                        "action_id": "feedback_negative",
                        "value": clean_text
                    }
                ]
            }
        ]

        say(
            text=answer_text,
            blocks=blocks,
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

        # メンションを含むメッセージは無視（app_mentionイベントで処理）
        text = event.get("text", "")
        if re.search(r'<@[A-Z0-9]+>', text):
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

        # RAGで回答を生成
        result = rag_service.answer_question(text)

        # 回答を整形
        answer_text = result['answer']

        # 参考ドキュメントがある場合は追記
        if result['sources']:
            answer_text += "\n\n_参考ドキュメント:_"
            for i, source in enumerate(result['sources'][:2], 1):  # 最大2件
                answer_text += f"\n• {source['title']}"

        # Block Kitでフィードバックボタン付きメッセージを送信
        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": answer_text}
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "_この回答は役に立ちましたか？_"}
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "👍 良い", "emoji": True},
                        "style": "primary",
                        "action_id": "feedback_positive",
                        "value": text
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "👎 改善が必要", "emoji": True},
                        "action_id": "feedback_negative",
                        "value": text
                    }
                ]
            }
        ]

        say(
            text=answer_text,
            blocks=blocks,
            thread_ts=thread_ts or ts
        )

        logger.info(f"Auto-response sent")

    except Exception as e:
        logger.error(f"Error handling message: {e}", exc_info=True)


@app.action("feedback_positive")
def handle_feedback_positive(ack, body, client):
    """
    👍ボタンクリック時の処理
    """
    ack()
    try:
        user = body["user"]["id"]
        channel = body["channel"]["id"]
        message = body["message"]
        message_ts = message["ts"]
        question = body["actions"][0].get("value", "")
        answer = message.get("text", "")

        feedback_logger.log_feedback(
            feedback_type="positive",
            question=question,
            answer=answer,
            channel=channel,
            user=user,
            message_ts=message_ts
        )

        # ボタンを「ありがとうございます」に更新
        blocks = message.get("blocks", [])[:-1]  # actionsブロックを削除
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": "✅ フィードバックありがとうございます！"}]
        })

        client.chat_update(
            channel=channel,
            ts=message_ts,
            text=answer,
            blocks=blocks
        )

        logger.info(f"Positive feedback recorded from {user}")

    except Exception as e:
        logger.error(f"Error handling positive feedback: {e}", exc_info=True)


@app.action("feedback_negative")
def handle_feedback_negative(ack, body, client):
    """
    👎ボタンクリック時の処理
    """
    ack()
    try:
        user = body["user"]["id"]
        channel = body["channel"]["id"]
        message = body["message"]
        message_ts = message["ts"]
        question = body["actions"][0].get("value", "")
        answer = message.get("text", "")

        feedback_logger.log_feedback(
            feedback_type="negative",
            question=question,
            answer=answer,
            channel=channel,
            user=user,
            message_ts=message_ts
        )

        # ボタンを「ありがとうございます」に更新
        blocks = message.get("blocks", [])[:-1]  # actionsブロックを削除
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": "✅ フィードバックありがとうございます！改善に活用します。"}]
        })

        client.chat_update(
            channel=channel,
            ts=message_ts,
            text=answer,
            blocks=blocks
        )

        logger.info(f"Negative feedback recorded from {user}")

    except Exception as e:
        logger.error(f"Error handling negative feedback: {e}", exc_info=True)


@app.event("reaction_added")
def handle_reaction_added(event, client):
    """
    リアクション追加イベントの処理
    👍(+1) / 👎(-1) でフィードバックを記録（後方互換）
    """
    try:
        reaction = event.get("reaction")
        user = event.get("user")
        item = event.get("item", {})
        channel = item.get("channel")
        message_ts = item.get("ts")

        # 対象のリアクションかチェック
        if reaction not in ["+1", "-1", "thumbsup", "thumbsdown"]:
            return

        # ボットのユーザーIDを取得
        bot_user_id = client.auth_test()["user_id"]

        # リアクションが付けられたメッセージを取得
        result = client.conversations_history(
            channel=channel,
            latest=message_ts,
            limit=1,
            inclusive=True
        )

        messages = result.get("messages", [])
        if not messages:
            return

        message = messages[0]

        # ボットのメッセージかチェック
        if message.get("user") != bot_user_id and not message.get("bot_id"):
            return

        # 元の質問を取得（スレッドの親メッセージ）
        thread_ts = message.get("thread_ts")
        question = ""

        if thread_ts:
            thread_result = client.conversations_replies(
                channel=channel,
                ts=thread_ts,
                limit=1
            )
            thread_messages = thread_result.get("messages", [])
            if thread_messages:
                question = thread_messages[0].get("text", "")

        # フィードバックタイプを判定
        feedback_type = "positive" if reaction in ["+1", "thumbsup"] else "negative"

        # フィードバックを記録
        feedback_logger.log_feedback(
            feedback_type=feedback_type,
            question=question,
            answer=message.get("text", ""),
            channel=channel,
            user=user,
            message_ts=message_ts
        )

        logger.info(f"Feedback recorded: {feedback_type} from {user}")

    except Exception as e:
        logger.error(f"Error handling reaction: {e}", exc_info=True)


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
