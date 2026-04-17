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
from src.evaluation.evaluator import evaluate_and_log
import logging
import re

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

URL_PATTERN = re.compile(r'https?://[^\s<>]+')


def extract_urls(text: str) -> list:
    slack_urls = re.findall(r'<(https?://[^|>]+)(?:\|[^>]*)?>',  text)
    plain_urls = URL_PATTERN.findall(text)
    return list(set(slack_urls + plain_urls))


def format_sources_section(sources_by_type: dict, is_unable: bool) -> str:
    """ソース別の参考ドキュメントセクションをフォーマット"""
    sections = []

    drive_sources = sources_by_type.get('drive', [])
    if drive_sources and not is_unable:
        section = f"*①ナレッジ(GoogleDrive)から類似事例を検索*\n{len(drive_sources)}件見つかりました。"
        for src in drive_sources[:3]:
            section += f"\n• <{src['link']}|{src['title']}>" if src.get('link') else f"\n• {src['title']}"
        sections.append(section)
    else:
        sections.append("*①ナレッジ(GoogleDrive)から類似事例を検索*\n該当なし")

    slack_sources = sources_by_type.get('slack', [])
    if slack_sources and not is_unable:
        section = f"*②Slackの過去事例*\n{len(slack_sources)}件見つかりました。"
        for src in slack_sources[:3]:
            section += f"\n• <{src['link']}|{src['title']}>" if src.get('link') else f"\n• {src['title']}"
        sections.append(section)
    else:
        sections.append("*②Slackの過去事例*\n該当なし")

    web_sources = sources_by_type.get('web', [])
    if web_sources:
        section = "*③Web検索の結果*"
        for src in web_sources[:3]:
            section += f"\n• <{src['url']}|{src['title']}>"
        sections.append(section)
    elif is_unable or (not drive_sources and not slack_sources):
        sections.append("*③Web検索の結果*\n該当なし")

    return "\n\n".join(sections)


def format_confidence_indicator(confidence_score: float, grounding_warning: str = None) -> str:
    if confidence_score >= 0.7:
        indicator = "🟢 高"
    elif confidence_score >= 0.4:
        indicator = "🟡 中"
    else:
        indicator = "🔴 低"

    text = f"*回答の信頼度:* {indicator} ({confidence_score:.0%})"
    if grounding_warning:
        text += f"\n⚠️ {grounding_warning}"
    return text


def _get_thread_reply_status(client, channel: str, thread_ts: str) -> tuple:
    """(has_human_reply, has_bot_reply) を返す"""
    try:
        bot_user_id = client.auth_test()["user_id"]
        result = client.conversations_replies(channel=channel, ts=thread_ts, limit=100)
        has_human = has_bot = False
        for msg in result.get("messages", [])[1:]:
            if msg.get("user") == bot_user_id or msg.get("bot_id"):
                has_bot = True
            else:
                has_human = True
        return has_human, has_bot
    except Exception as e:
        logger.error(f"Error checking thread replies: {e}")
        return False, False


def _build_reply_blocks(answer_text: str, question_text: str) -> list:
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": answer_text}},
        {"type": "divider"},
        {"type": "section", "text": {"type": "mrkdwn", "text": "_この回答は役に立ちましたか？_"}},
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "👍 良い", "emoji": True},
                    "style": "primary",
                    "action_id": "feedback_positive",
                    "value": question_text
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "👎 改善が必要", "emoji": True},
                    "action_id": "feedback_negative",
                    "value": question_text
                }
            ]
        }
    ]


def _answer_and_reply(text: str, channel: str, thread_ts: str, user: str, event: dict, say):
    """質問への回答と返信を処理する共通ロジック"""
    images = fetch_images_from_event(event, settings.slack_bot_token)

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
        blocks=_build_reply_blocks(answer_text, text),
        thread_ts=thread_ts
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
        if event.get("subtype") or event.get("bot_id"):
            return

        text = event.get("text", "")
        if re.search(r'<@[A-Z0-9]+>', text):
            return

        channel = event.get("channel")
        if channel not in AUTO_REPLY_CHANNELS:
            return

        thread_ts = event.get("thread_ts")
        ts = event.get("ts")

        if thread_ts:
            has_human, has_bot = _get_thread_reply_status(client, channel, thread_ts)
            if has_human:
                logger.info(f"Skipping: Human has already replied to thread {thread_ts}")
                return
            if has_bot:
                logger.info(f"Skipping: Bot has already replied to thread {thread_ts}")
                return

        logger.info(f"Auto-replying to message from {event.get('user')} in channel {channel}: {text}")
        _answer_and_reply(text, channel, thread_ts or ts, event.get("user"), event, say)

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

    load_slack_history_on_startup()

    handler = SocketModeHandler(app, settings.slack_app_token)
    handler.start()


if __name__ == "__main__":
    start_bot()
