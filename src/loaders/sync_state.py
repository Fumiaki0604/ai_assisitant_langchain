"""
差分インデックス用の状態管理
S3_STATE_BUCKET 環境変数が設定されている場合はS3に保存（ECS用）
未設定の場合はローカルファイル（ローカル開発用）
"""
import json
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_KEY = "sync_state.json"
LOCAL_STATE_FILE = "data/sync_state.json"


def _get_s3_bucket() -> str | None:
    return os.environ.get("S3_STATE_BUCKET")


def load_state() -> dict:
    bucket = _get_s3_bucket()
    if bucket:
        return _load_from_s3(bucket)
    return _load_from_local()


def save_state(state: dict):
    bucket = _get_s3_bucket()
    if bucket:
        _save_to_s3(bucket, state)
    else:
        _save_to_local(state)


def _load_from_s3(bucket: str) -> dict:
    try:
        import boto3
        from botocore.exceptions import ClientError
        s3 = boto3.client("s3")
        response = s3.get_object(Bucket=bucket, Key=STATE_KEY)
        return json.loads(response["Body"].read().decode("utf-8"))
    except Exception as e:
        # NoSuchKey の場合は初回起動として扱う
        if "NoSuchKey" in str(e):
            logger.info("No existing sync state in S3, starting fresh.")
        else:
            logger.warning(f"Failed to load state from S3: {e}. Starting fresh.")
        return {"slack": {}, "google_drive": {}, "notion": {}}


def _save_to_s3(bucket: str, state: dict):
    try:
        import boto3
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=bucket,
            Key=STATE_KEY,
            Body=json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json"
        )
        logger.info(f"Sync state saved to s3://{bucket}/{STATE_KEY}")
    except Exception as e:
        logger.error(f"Failed to save state to S3: {e}")


def _load_from_local() -> dict:
    if os.path.exists(LOCAL_STATE_FILE):
        try:
            with open(LOCAL_STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load sync state: {e}. Starting fresh.")
    return {"slack": {}, "google_drive": {}, "notion": {}}


def _save_to_local(state: dict):
    Path(LOCAL_STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(LOCAL_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def delete_vectors(vector_ids: list, index_name: str) -> list:
    """
    Pineconeから指定IDのベクターを削除。
    Returns: 削除に失敗したIDのリスト（空 = 全件成功）
    """
    if not vector_ids:
        return []
    failed = []
    try:
        from pinecone import Pinecone
        from config.settings import settings
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(index_name)
        for i in range(0, len(vector_ids), 100):
            batch = vector_ids[i:i + 100]
            try:
                index.delete(ids=batch)
            except Exception as e:
                logger.error(f"Failed to delete batch ({len(batch)} IDs): {e}")
                failed.extend(batch)
        if not failed:
            logger.info(f"Deleted {len(vector_ids)} vectors from Pinecone")
        else:
            logger.warning(f"Deleted {len(vector_ids) - len(failed)}/{len(vector_ids)} vectors. {len(failed)} failed.")
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone for deletion: {e}")
        failed = list(vector_ids)
    return failed


def save_docs_with_ids(documents: list, id_prefixes: list, text_splitter, index_name: str) -> list:
    """
    決定論的IDでPineconeにHybrid Search（dense + sparse BM25）で保存。

    documents: list of {"content": str, "metadata": dict} または LangChain Document
    id_prefixes: list of str, ドキュメントごとに1つ
    Returns: list of list of vector_ids（ドキュメントごとのベクターIDリスト）
    """
    from pinecone import Pinecone
    from pinecone_text.sparse import BM25Encoder
    from src.rag.embeddings import get_embeddings
    from config.settings import settings as _settings

    embeddings = get_embeddings()
    all_texts = []
    all_metadatas = []
    all_ids = []
    doc_id_lists = [[] for _ in documents]

    for doc_i, (doc, prefix) in enumerate(zip(documents, id_prefixes)):
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
        bm25 = BM25Encoder().default()
        dense_vectors = embeddings.embed_documents(all_texts)
        sparse_vectors = bm25.encode_documents(all_texts)

        vectors = [
            {
                "id": all_ids[i],
                "values": dense_vectors[i],
                "sparse_values": sparse_vectors[i],
                "metadata": {**all_metadatas[i], "text": all_texts[i]},
            }
            for i in range(len(all_texts))
        ]

        pc = Pinecone(api_key=_settings.pinecone_api_key)
        index = pc.Index(index_name)
        for batch_start in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[batch_start:batch_start + 100])

        logger.info(f"Saved {len(all_texts)} chunks with hybrid vectors (dense+sparse) to Pinecone")

    return doc_id_lists
