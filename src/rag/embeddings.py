"""
Ruri v3-310m (cl-nagoya) を使用した埋め込み生成
日本語特化・768次元・CPU推論

旧: Amazon Titan Text Embeddings V2 (Bedrock, 1024次元)
新: Ruri v3-310m (HuggingFace, 768次元)
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from langchain_community.embeddings import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)

RURI_MODEL_NAME = "cl-nagoya/ruri-v3-310m"

# モデルロードは重いためシングルトンでキャッシュする
_embeddings_instance = None


def get_embeddings():
    """
    Ruri v3-310m の埋め込みモデルを取得（シングルトン）

    ModernBERT-Jaベース・日本語特化・768次元
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=RURI_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"Ruri embeddings initialized: {RURI_MODEL_NAME}")
    return _embeddings_instance
