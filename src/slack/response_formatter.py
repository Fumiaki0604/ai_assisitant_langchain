"""Slack 返答のフォーマット処理"""


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
            # 生URLを使うことでSlackのスレッドプレビューカード(unfurl)を表示
            section += f"\n{src['link']}" if src.get('link') else f"\n• {src['title']}"
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


def build_reply_blocks(answer_text: str, question_text: str) -> list:
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
