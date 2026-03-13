"""
差分インデックス用の状態管理
data/sync_state.json にインデックス済みドキュメントの情報を保存
"""
import json
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_FILE = "data/sync_state.json"


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load sync state: {e}. Starting fresh.")
    return {"slack": {}, "google_drive": {}, "notion": {}}


def save_state(state: dict):
    Path(STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def delete_vectors(vector_ids: list, index_name: str):
    """Pineconeから指定IDのベクターを削除"""
    if not vector_ids:
        return
    try:
        from pinecone import Pinecone
        from config.settings import settings
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(index_name)
        # Pineconeはバッチ削除に上限があるため100件ずつ
        for i in range(0, len(vector_ids), 100):
            index.delete(ids=vector_ids[i:i + 100])
        logger.info(f"Deleted {len(vector_ids)} vectors from Pinecone")
    except Exception as e:
        logger.error(f"Failed to delete vectors: {e}")


def save_docs_with_ids(documents: list, id_prefixes: list, text_splitter, index_name: str) -> list:
    """
    決定論的IDでPineconeに保存。

    documents: list of {"content": str, "metadata": dict} または LangChain Document
    id_prefixes: list of str, ドキュメントごとに1つ
    Returns: list of list of vector_ids（ドキュメントごとのベクターIDリスト）
    """
    from langchain_pinecone import PineconeVectorStore
    from src.rag.embeddings import get_embeddings

    embeddings = get_embeddings()
    all_texts = []
    all_metadatas = []
    all_ids = []
    doc_id_lists = [[] for _ in documents]

    for doc_i, (doc, prefix) in enumerate(zip(documents, id_prefixes)):
        # LangChain Document と dict の両形式に対応
        if hasattr(doc, 'page_content'):
            content = doc.page_content
            metadata = doc.metadata.copy()
        else:
            content = doc["content"]
            metadata = doc["metadata"].copy()

        chunks = text_splitter.split_text(content)
        for chunk_i, chunk in enumerate(chunks):
            vec_id = f"{prefix}_{chunk_i}"
            meta = metadata.copy()
            meta["chunk_id"] = chunk_i
            all_texts.append(chunk)
            all_metadatas.append(meta)
            all_ids.append(vec_id)
            doc_id_lists[doc_i].append(vec_id)

    if all_texts:
        PineconeVectorStore.from_texts(
            texts=all_texts,
            embedding=embeddings,
            metadatas=all_metadatas,
            ids=all_ids,
            index_name=index_name
        )
        logger.info(f"Saved {len(all_texts)} chunks with deterministic IDs to Pinecone")

    return doc_id_lists
