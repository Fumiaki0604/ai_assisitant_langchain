"""
Pineconeインデックスを削除して再作成
"""
import sys
import os

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pinecone import Pinecone
from config.settings import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reset_pinecone_index():
    """
    既存のインデックスを削除して再作成
    """
    try:
        pc = Pinecone(api_key=settings.pinecone_api_key)

        # 既存のインデックスを確認
        existing_indexes = pc.list_indexes()
        index_names = [index['name'] for index in existing_indexes]

        if settings.pinecone_index_name in index_names:
            logger.info(f"インデックス '{settings.pinecone_index_name}' を削除中...")
            pc.delete_index(settings.pinecone_index_name)
            logger.info(f"✅ インデックスを削除しました")
        else:
            logger.info(f"インデックス '{settings.pinecone_index_name}' は存在しません")

        print("\n削除完了！次に setup_pinecone.py を実行してインデックスを再作成してください。\n")
        print("python scripts/setup_pinecone.py\n")

        return True

    except Exception as e:
        logger.error(f"エラー: {e}", exc_info=True)
        print(f"\n❌ エラー: {e}\n")
        return False


if __name__ == "__main__":
    reset_pinecone_index()
