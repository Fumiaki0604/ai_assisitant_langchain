"""
Pineconeデータ差分同期スクリプト
新規・更新されたドキュメントのみをPineconeに反映する（差分インデックス）
"""
import sys
import os
import re
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.loaders.slack_loader import SlackHistoryLoader
from src.loaders.google_drive_loader import GoogleDriveLoader
from src.loaders.notion_loader import NotionLoader
from src.loaders.sync_state import load_state, save_state, delete_vectors, save_docs_with_ids
from config.settings import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def _safe_id(s: str) -> str:
    """Pinecone IDとして使用可能な文字列に変換"""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', s)


def _flush_pending_deletes(state: dict):
    """前回削除に失敗したベクターIDを再試行"""
    pending = state.get("pending_delete_ids", [])
    if not pending:
        return
    logger.info(f"Retrying deletion of {len(pending)} pending vector IDs...")
    failed = delete_vectors(pending, settings.pinecone_index_name)
    if failed:
        state["pending_delete_ids"] = failed
        logger.warning(f"{len(failed)} IDs still pending deletion, will retry next run")
    else:
        state.pop("pending_delete_ids", None)
        logger.info("All pending deletions cleared")


def _queue_delete(state: dict, old_ids: list):
    """削除対象IDを処理し、失敗分をpending_delete_idsに退避"""
    if not old_ids:
        return
    failed = delete_vectors(old_ids, settings.pinecone_index_name)
    if failed:
        state.setdefault("pending_delete_ids", []).extend(failed)
        logger.warning(f"{len(failed)} IDs queued for retry deletion")


def sync_slack_history(state: dict) -> int:
    """Slack履歴を差分同期（reply_count変化時のみ更新）"""
    auto_reply = [ch.strip() for ch in settings.slack_auto_reply_channels.split(',') if ch.strip()]
    knowledge = [ch.strip() for ch in getattr(settings, 'slack_knowledge_channels', '').split(',') if ch.strip()]
    channels = list(dict.fromkeys(auto_reply + knowledge))

    if not channels:
        logger.info("No Slack channels configured")
        return 0

    slack_state = state.setdefault("slack", {})
    loader = SlackHistoryLoader()
    total_new = 0

    for channel_id in channels:
        try:
            channel_name = loader.get_channel_name(channel_id)
            messages = loader.fetch_channel_history(channel_id, limit=200)

            new_docs = []
            new_prefixes = []
            keys_to_update = []
            old_ids_to_delete = []

            for msg in messages:
                thread_ts = msg.get("thread_ts")
                ts = msg.get("ts", "")

                if thread_ts and thread_ts == ts:
                    # スレッド親メッセージ：reply_count変化時のみ更新
                    reply_count = msg.get("reply_count", 0)
                    state_key = f"{channel_id}_{ts.replace('.', '_')}"
                    existing = slack_state.get(state_key, {})

                    if existing.get("reply_count") == reply_count:
                        continue  # 変更なし

                    thread_messages = loader.fetch_thread_replies(channel_id, thread_ts)
                    if len(thread_messages) <= 1:
                        continue

                    doc = loader.format_thread_as_document(channel_id, channel_name, thread_messages)
                    if not doc:
                        continue

                    if existing.get("vector_ids"):
                        old_ids_to_delete.extend(existing["vector_ids"])

                    prefix = _safe_id(f"slack_{state_key}")
                    new_docs.append(doc)
                    new_prefixes.append(prefix)
                    keys_to_update.append((state_key, {"reply_count": reply_count}))

                elif not msg.get("thread_ts"):
                    # スレッドなし単独メッセージ：初回のみインデックス
                    text = msg.get("text", "")
                    if not text or len(text) <= 20:
                        continue

                    state_key = f"{channel_id}_solo_{ts.replace('.', '_')}"
                    if state_key in slack_state:
                        continue  # 既存

                    user = loader.get_user_name(msg.get("user", "Unknown"))
                    try:
                        dt = datetime.fromtimestamp(float(ts))
                        date_str = dt.strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        date_str = "Unknown"

                    permalink = loader.get_permalink(channel_id, ts)
                    doc = {
                        "content": f"投稿 ({user}): {text}",
                        "metadata": {
                            "source": "slack",
                            "channel_id": channel_id,
                            "channel_name": channel_name,
                            "thread_ts": ts,
                            "date": date_str,
                            "permalink": permalink,
                            "title": f"Slack: {text[:50]}..." if len(text) > 50 else f"Slack: {text}"
                        }
                    }
                    prefix = _safe_id(f"slack_{state_key}")
                    new_docs.append(doc)
                    new_prefixes.append(prefix)
                    keys_to_update.append((state_key, {}))

            if old_ids_to_delete:
                _queue_delete(state, old_ids_to_delete)

            if new_docs:
                vector_ids_map = save_docs_with_ids(
                    new_docs, new_prefixes, TEXT_SPLITTER, settings.pinecone_index_name
                )
                for i, (state_key, base_info) in enumerate(keys_to_update):
                    entry = dict(base_info)
                    entry["vector_ids"] = vector_ids_map[i]
                    slack_state[state_key] = entry

                total_new += len(new_docs)
                logger.info(f"Synced {len(new_docs)} new/updated docs from #{channel_name}")
            else:
                logger.info(f"No changes in #{channel_name}")

        except Exception as e:
            logger.error(f"Failed to sync Slack channel {channel_id}: {e}")

    state["slack"] = slack_state
    return total_new


def sync_google_drive(state: dict) -> int:
    """Google Driveを差分同期（modifiedTime変化時のみ更新）"""
    folder_id = getattr(settings, 'google_drive_folder_id', None)
    gdrive_state = state.setdefault("google_drive", {})

    try:
        loader = GoogleDriveLoader()
        files = loader.list_files_in_folder(folder_id)

        new_docs = []
        new_prefixes = []
        keys_to_update = []
        old_ids_to_delete = []

        for file_info in files:
            file_id = file_info["id"]
            modified_time = file_info.get("modifiedTime", "")
            existing = gdrive_state.get(file_id, {})

            if existing.get("modified_time") == modified_time:
                continue  # 変更なし

            doc = loader.load_file(file_info)
            if not doc:
                continue

            # 文字化けチェック（GoogleDriveLoader固有）
            cleaned = loader._clean_text(doc.page_content)
            if loader._is_garbled(cleaned):
                logger.warning(f"Skipping garbled file: {file_info.get('name')}")
                continue

            doc = Document(page_content=cleaned, metadata=doc.metadata)

            if existing.get("vector_ids"):
                old_ids_to_delete.extend(existing["vector_ids"])

            prefix = _safe_id(f"gdrive_{file_id}")
            new_docs.append(doc)
            new_prefixes.append(prefix)
            keys_to_update.append((file_id, modified_time))

        if old_ids_to_delete:
            _queue_delete(state, old_ids_to_delete)

        if new_docs:
            vector_ids_map = save_docs_with_ids(
                new_docs, new_prefixes, TEXT_SPLITTER, settings.pinecone_index_name
            )
            for i, (file_id, modified_time) in enumerate(keys_to_update):
                gdrive_state[file_id] = {
                    "modified_time": modified_time,
                    "vector_ids": vector_ids_map[i]
                }
            state["google_drive"] = gdrive_state
            logger.info(f"Synced {len(new_docs)} new/updated files from Google Drive")
            return len(new_docs)
        else:
            logger.info("No changes in Google Drive")
            return 0

    except Exception as e:
        logger.error(f"Failed to sync Google Drive: {e}")
        return 0


def sync_notion(state: dict) -> int:
    """Notionを差分同期（last_edited_time変化時のみ更新）"""
    notion_api_key = getattr(settings, 'notion_api_key', None)
    if not notion_api_key:
        logger.info("NOTION_API_KEY not configured, skipping Notion sync")
        return 0

    notion_state = state.setdefault("notion", {})

    try:
        loader = NotionLoader()
        all_pages = loader.search_pages()

        new_docs = []
        new_prefixes = []
        keys_to_update = []
        old_ids_to_delete = []

        for page in all_pages:
            page_id = page.get("id", "")
            last_edited = page.get("last_edited_time", "")
            existing = notion_state.get(page_id, {})

            if existing.get("last_edited_time") == last_edited:
                continue  # 変更なし

            title = loader.get_page_title(page)
            url = page.get("url", "")
            content = loader.get_page_content(page_id)

            pdf_urls = loader._extract_pdf_urls_from_properties(page)
            pdf_texts = [
                t for pdf_url in pdf_urls
                for t in [loader._extract_text_from_pdf_url(pdf_url, title)]
                if t
            ]
            if pdf_texts:
                pdf_block = "\n\n".join(pdf_texts)
                content = f"{content}\n\n{pdf_block}" if content else pdf_block

            if not content:
                continue

            doc = {
                "content": f"# {title}\n\n{content}",
                "metadata": {
                    "source": "notion",
                    "page_id": page_id,
                    "title": f"Notion: {title}",
                    "url": url,
                    "pdf_count": len(pdf_urls)
                }
            }

            if existing.get("vector_ids"):
                old_ids_to_delete.extend(existing["vector_ids"])

            prefix = f"notion_{page_id.replace('-', '')}"
            new_docs.append(doc)
            new_prefixes.append(prefix)
            keys_to_update.append((page_id, last_edited))

        if old_ids_to_delete:
            _queue_delete(state, old_ids_to_delete)

        if new_docs:
            vector_ids_map = save_docs_with_ids(
                new_docs, new_prefixes, TEXT_SPLITTER, settings.pinecone_index_name
            )
            for i, (page_id, last_edited) in enumerate(keys_to_update):
                notion_state[page_id] = {
                    "last_edited_time": last_edited,
                    "vector_ids": vector_ids_map[i]
                }
            state["notion"] = notion_state
            logger.info(f"Synced {len(new_docs)} new/updated pages from Notion")
            return len(new_docs)
        else:
            logger.info("No changes in Notion")
            return 0

    except Exception as e:
        logger.error(f"Failed to sync Notion: {e}")
        return 0


def main():
    logger.info("Starting incremental Pinecone sync...")
    state = load_state()

    _flush_pending_deletes(state)

    slack_count = sync_slack_history(state)
    drive_count = sync_google_drive(state)
    notion_count = sync_notion(state)

    save_state(state)
    logger.info(
        f"Sync completed: Slack={slack_count} new/updated, "
        f"GoogleDrive={drive_count} new/updated, "
        f"Notion={notion_count} new/updated"
    )


if __name__ == "__main__":
    main()
