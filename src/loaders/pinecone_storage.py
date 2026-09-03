"""Pinecone へのドキュメント保存（全ローダー共通）"""
import os
import logging

from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.embeddings import get_embeddings
from config.settings import settings

logger = logging.getLogger(__name__)

_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)


def save_to_pinecone(documents: list) -> bool:
    """
    ドキュメントリストを Pinecone に保存する。

    documents の各要素は以下のいずれかの形式に対応:
      - LangChain Document オブジェクト (doc.page_content, doc.metadata)
      - dict {"content": str, "metadata": dict}
    """
    if not documents:
        logger.warning("No documents to save")
        return False

    os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key

    try:
        embeddings = get_embeddings()
        all_texts = []
        all_metadatas = []

        for doc in documents:
            if hasattr(doc, "page_content"):
                content = doc.page_content
                metadata = doc.metadata.copy()
            else:
                content = doc["content"]
                metadata = doc["metadata"].copy()

            chunks = _TEXT_SPLITTER.split_text(content)
            for i, chunk in enumerate(chunks):
                all_texts.append(chunk)
                chunk_meta = metadata.copy()
                chunk_meta["chunk_id"] = i
                all_metadatas.append(chunk_meta)

        logger.info(f"Saving {len(all_texts)} chunks to Pinecone...")
        PineconeVectorStore.from_texts(
            texts=all_texts,
            embedding=embeddings,
            metadatas=all_metadatas,
            index_name=settings.pinecone_index_name,
        )
        logger.info("Documents saved to Pinecone successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to save to Pinecone: {e}", exc_info=True)
        return False
