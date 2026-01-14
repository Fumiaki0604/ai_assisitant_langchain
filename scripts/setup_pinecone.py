"""
Pineconeインデックスのセットアップと接続テスト
"""
import sys
import os

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pinecone import Pinecone, ServerlessSpec
from config.settings import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_pinecone_index():
    """
    Pineconeインデックスを作成またはテスト
    """
    try:
        # Pineconeクライアントを初期化
        pc = Pinecone(api_key=settings.pinecone_api_key)

        logger.info("Pinecone接続成功")
        logger.info(f"インデックス名: {settings.pinecone_index_name}")

        # 既存のインデックスを確認
        existing_indexes = pc.list_indexes()
        index_names = [index['name'] for index in existing_indexes]

        logger.info(f"既存のインデックス: {index_names}")

        if settings.pinecone_index_name not in index_names:
            logger.info(f"インデックス '{settings.pinecone_index_name}' を作成中...")

            # インデックスを作成（Serverless Starter用）
            pc.create_index(
                name=settings.pinecone_index_name,
                dimension=1024,  # Bedrock Embeddings (Titan Text Embeddings V2) の次元数
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )

            logger.info(f"✅ インデックス '{settings.pinecone_index_name}' を作成しました")
        else:
            logger.info(f"✅ インデックス '{settings.pinecone_index_name}' は既に存在します")

        # インデックスに接続してテスト
        index = pc.Index(settings.pinecone_index_name)
        stats = index.describe_index_stats()

        logger.info(f"インデックス統計: {stats}")
        logger.info(f"ベクトル数: {stats.get('total_vector_count', 0)}")

        print("\n" + "="*50)
        print("✅ Pineconeセットアップ完了！")
        print(f"インデックス名: {settings.pinecone_index_name}")
        print(f"ベクトル数: {stats.get('total_vector_count', 0)}")
        print("="*50 + "\n")

        return True

    except Exception as e:
        logger.error(f"Pineconeセットアップエラー: {e}", exc_info=True)
        print(f"\n❌ エラー: {e}\n")
        return False


if __name__ == "__main__":
    setup_pinecone_index()
