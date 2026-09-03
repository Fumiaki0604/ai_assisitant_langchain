"""
Slackボットのメイン実装
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from config.settings import settings
from src.rag.rag_service import get_rag_service
from src.rag.web_searcher import fetch_url_content
from src.feedback.feedback_logger import get_feedback_logger
from src.loaders.slack_loader import SlackHistoryLoader
from src.slack.image_handler import fetch_images_from_event
from src.slack.response_formatter import format_sources_section, format_confidence_indicator, build_reply_blocks
from src.evaluation.evaluator import evaluate_and_log
import logging
import re
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = App(
    token=settings.slack_bot_token,
    signing_secret=settings.slack_signing_secret
)

rag_service = get_rag_service()
feedback_logger = get_feedback_logger()

AUTO_REPLY_CHANNELS = [ch.strip() for ch in settings.slack_auto_reply_channels.split(',') if ch.strip()]
NO_WEB_SEARCH_CHANNELS = [ch.strip() for ch in settings.slack_no_web_search_channels.split(',') if ch.strip()]

# ボット自身のユーザーID（起動時に取得してキャッシュ）
_BOT_USER_ID: str = None


def _get_bot_user_id(client) -> str:
    global _BOT_USER_ID
    if _BOT_USER_ID is None:
        _BOT_USER_ID = client.auth_test()["user_id"]
    return _BOT_USER_ID

URL_PATTERN = re.compile(r'https?://[^\s<>]+')


def extract_urls(text: str) -> list:
    slack_urls = re.findall(r'<(https?://[^|>]+)(?:\|[^>]*)?>',  text)
    plain_urls = URL_PATTERN.findall(text)
    return list(set(slack_urls + plain_urls))


def _answer_and_reply(text: str, channel: str, thread_ts: str, user: str, event: dict, say, classify_intent: bool = True):
    """質問への回答と返信を処理する共通ロジック"""
    images = fetch_images_from_event(event, settings.slack_bot_token)

    if classify_intent:
        intent_result = rag_service.classify_message_intent(text)
        if intent_result["intent"] == "share":
            say(text=intent_result["acknowledgment"], thread_ts=thread_ts)
            logger.info(f"Acknowledgment sent (share message)")
            return

    url_content = ""
    urls = extract_urls(text)
    if urls:
        logger.info(f"Fetching URL: {urls[0]}")
        url_content = fetch_url_content(urls[0])

    result = rag_service.answer_question(
        text, url_content, images=images,
        skip_web_search=(channel in NO_WEB_SEARCH_CHANNELS)
    )

    answer_text = result['answer']

    sources_section = format_sources_section(
        result.get('sources_by_type', {}),
        result.get('is_unable_to_answer', False)
    )
    if sources_section:
        answer_text += "\n\n---\n*参考情報*\n" + sources_section

    if not result.get('is_unable_to_answer', False):
        answer_text += "\n\n" + format_confidence_indicator(
            result.get('confidence_score', 0.0),
            result.get('grounding_warning')
        )

    say(
        text=answer_text,
        blocks=build_reply_blocks(answer_text, text),
        thread_ts=thread_ts,
        unfurl_links=True,
    )
    logger.info(f"Response sent to {user}")

    evaluate_and_log(
        question=text,
        answer=result['answer'],
        question_type=result.get('question_type', 'knowledge'),
        channel=channel,
        user=user,
        thread_ts=thread_ts,
    )


@app.event("app_mention")
def handle_mention(event, say, client):
    try:
        text = re.sub(r'<@[A-Z0-9]+>', '', event["text"]).strip()
        thread_ts = event.get("thread_ts") or event["ts"]
        logger.info(f"Received mention from {event['user']}: {text}")
        _answer_and_reply(text, event["channel"], thread_ts, event["user"], event, say)
    except Exception as e:
        logger.error(f"Error handling mention: {e}", exc_info=True)
        say(
            text=f"申し訳ございません。エラーが発生しました: {str(e)}",
            thread_ts=event.get("thread_ts") or event.get("ts")
        )


@app.event("message")
def handle_message_events(event, say, client):
    try:
        subtype = event.get("subtype")
        if (subtype and subtype != "file_share") or event.get("bot_id"):
            return

        text = event.get("text", "")
        # ボット自身へのメンションのみスキップ（app_mentionで処理）
        # 他ユーザーへのメンションはスキップしない
        bot_user_id = _get_bot_user_id(client)
        if f'<@{bot_user_id}>' in text:
            return

        channel = event.get("channel")
        if channel not in AUTO_REPLY_CHANNELS:
            return

        thread_ts = event.get("thread_ts")
        ts = event.get("ts")

        if thread_ts:
            # スレッド返信の場合: スレッドのコンテキストを取得
            try:
                result = client.conversations_replies(channel=channel, ts=thread_ts, limit=100)
                messages = result.get("messages", [])
                if not messages:
                    return

                bot_user_id = _get_bot_user_id(client)
                original_poster = messages[0].get("user")
                current_user = event.get("user")

                # 元の質問者以外の返信（専門家回答など）はスキップ
                if current_user != original_poster:
                    logger.info(f"Skipping: reply from {current_user} (not original poster {original_poster})")
                    return

                replies = messages[1:]
                # ボットが既に返信済みの場合はスキップ
                has_bot = any(m.get("user") == bot_user_id or m.get("bot_id") for m in replies)
                if has_bot:
                    logger.info(f"Skipping: Bot already replied to thread {thread_ts}")
                    return

                # 専門家（元の質問者以外の人間）が既に関与している場合はスキップ
                has_human_expert = any(
                    m.get("user") and m.get("user") != bot_user_id and not m.get("bot_id") and m.get("user") != original_poster
                    for m in replies
                )
                if has_human_expert:
                    logger.info(f"Skipping: Human expert already engaged in thread {thread_ts}")
                    return
            except Exception as e:
                logger.error(f"Error checking thread context: {e}")
                return

        logger.info(f"Auto-replying to message from {event.get('user')} in channel {channel}: {text}")
        # ルートメッセージはintent分類をスキップ（常に質問として処理）
        # スレッド返信のみ分類（「ありがとうございます」等を除外するため）
        is_root_message = thread_ts is None
        _answer_and_reply(text, channel, thread_ts or ts, event.get("user"), event, say, classify_intent=not is_root_message)

    except Exception as e:
        logger.error(f"Error handling message: {e}", exc_info=True)


def _handle_feedback(ack, body, client, feedback_type: str, thank_msg: str):
    ack()
    try:
        user = body["user"]["id"]
        channel = body["channel"]["id"]
        message = body["message"]
        message_ts = message["ts"]
        question = body["actions"][0].get("value", "")
        answer = message.get("text", "")

        feedback_logger.log_feedback(
            feedback_type=feedback_type,
            question=question,
            answer=answer,
            channel=channel,
            user=user,
            message_ts=message_ts
        )

        blocks = message.get("blocks", [])[:-1]
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": thank_msg}]
        })
        client.chat_update(channel=channel, ts=message_ts, text=answer, blocks=blocks)
        logger.info(f"{feedback_type} feedback recorded from {user}")

    except Exception as e:
        logger.error(f"Error handling {feedback_type} feedback: {e}", exc_info=True)


@app.action("feedback_positive")
def handle_feedback_positive(ack, body, client):
    _handle_feedback(ack, body, client, "positive", "✅ フィードバックありがとうございます！")


@app.action("feedback_negative")
def handle_feedback_negative(ack, body, client):
    _handle_feedback(ack, body, client, "negative", "✅ フィードバックありがとうございます！改善に活用します。")


@app.event("reaction_added")
def handle_reaction_added(event, client):
    """リアクション追加イベントの処理（👍/👎フィードバック記録）"""
    try:
        reaction = event.get("reaction")
        if reaction not in ["+1", "-1", "thumbsup", "thumbsdown"]:
            return

        user = event.get("user")
        item = event.get("item", {})
        channel = item.get("channel")
        message_ts = item.get("ts")

        bot_user_id = client.auth_test()["user_id"]
        result = client.conversations_history(channel=channel, latest=message_ts, limit=1, inclusive=True)
        messages = result.get("messages", [])
        if not messages:
            return

        message = messages[0]
        if message.get("user") != bot_user_id and not message.get("bot_id"):
            return

        question = ""
        thread_ts = message.get("thread_ts")
        if thread_ts:
            thread_result = client.conversations_replies(channel=channel, ts=thread_ts, limit=1)
            thread_messages = thread_result.get("messages", [])
            if thread_messages:
                question = thread_messages[0].get("text", "")

        feedback_type = "positive" if reaction in ["+1", "thumbsup"] else "negative"
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


def answer_missed_questions(slack_client, lookback_minutes: int = 60):
    """起動時に未回答の質問に遡って回答する"""
    if not AUTO_REPLY_CHANNELS:
        return

    import time
    oldest = str(time.time() - lookback_minutes * 60)
    bot_user_id = slack_client.auth_test()["user_id"]

    for channel_id in AUTO_REPLY_CHANNELS:
        try:
            result = slack_client.conversations_history(channel=channel_id, oldest=oldest, limit=50)
            messages = result.get("messages", [])
            # 古い順に処理
            for msg in reversed(messages):
                if msg.get("subtype") or msg.get("bot_id"):
                    continue
                if f'<@{bot_user_id}>' in msg.get("text", ""):
                    continue

                ts = msg.get("ts")
                # スレッドの状態確認
                thread_result = slack_client.conversations_replies(channel=channel_id, ts=ts, limit=20)
                thread_msgs = thread_result.get("messages", [])

                has_bot = any(
                    m.get("user") == bot_user_id or m.get("bot_id")
                    for m in thread_msgs[1:]
                )
                if has_bot:
                    continue  # 既に回答済み

                logger.info(f"Answering missed question in {channel_id}: {msg.get('text', '')[:80]}")
                try:
                    # Slack clientからsayの代わりに直接投稿
                    text = msg.get("text", "")
                    images = fetch_images_from_event(msg, settings.slack_bot_token)
                    intent_result = rag_service.classify_message_intent(text)
                    if intent_result["intent"] == "share":
                        continue  # 共有メッセージはスキップ

                    result_qa = rag_service.answer_question(
                        text, "", images=images,
                        skip_web_search=(channel_id in NO_WEB_SEARCH_CHANNELS)
                    )
                    answer_text = result_qa['answer']
                    sources_section = format_sources_section(
                        result_qa.get('sources_by_type', {}),
                        result_qa.get('is_unable_to_answer', False)
                    )
                    if sources_section:
                        answer_text += "\n\n---\n*参考情報*\n" + sources_section
                    if not result_qa.get('is_unable_to_answer', False):
                        answer_text += "\n\n" + format_confidence_indicator(
                            result_qa.get('confidence_score', 0.0),
                            result_qa.get('grounding_warning')
                        )

                    slack_client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=ts,
                        text=answer_text,
                        blocks=build_reply_blocks(answer_text, text),
                        unfurl_links=True,
                    )
                    logger.info(f"Answered missed question ts={ts}")
                except Exception as e:
                    logger.error(f"Error answering missed question ts={ts}: {e}")
        except Exception as e:
            logger.error(f"Failed to check missed questions for channel {channel_id}: {e}")


def load_slack_history_on_startup():
    if not AUTO_REPLY_CHANNELS:
        logger.info("No auto-reply channels configured, skipping history load")
        return

    logger.info(f"Loading Slack history for channels: {AUTO_REPLY_CHANNELS}")
    loader = SlackHistoryLoader()
    for channel_id in AUTO_REPLY_CHANNELS:
        try:
            documents = loader.load_channel_threads(channel_id, limit=200)
            if documents:
                loader.save_to_pinecone(documents)
                logger.info(f"Loaded {len(documents)} documents from channel {channel_id}")
        except Exception as e:
            logger.error(f"Failed to load history for channel {channel_id}: {e}")


def start_bot():
    logger.info("Starting Slack bot...")
    logger.info(f"Bot token: {settings.slack_bot_token[:20]}...")
    logger.info(f"App token: {settings.slack_app_token[:20]}...")
    logger.info(f"Auto-reply channels: {AUTO_REPLY_CHANNELS}")

    # 起動時同期はバックグラウンドで実行（Socket Mode開始をブロックしない）
    threading.Thread(target=load_slack_history_on_startup, daemon=True).start()

    # 起動時に未回答の質問に遡って回答（デプロイ中の欠落を補完）
    slack_client = app.client
    threading.Thread(
        target=answer_missed_questions,
        args=(slack_client,),
        kwargs={"lookback_minutes": 60},
        daemon=True
    ).start()

    handler = SocketModeHandler(app, settings.slack_app_token)
    handler.start()


if __name__ == "__main__":
    start_bot()
