"""
AWS Bedrock Embeddingsを使用した埋め込み生成
"""
import sys
import os

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from langchain_aws import BedrockEmbeddings
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


def get_embeddings():
    """
    AWS Bedrock Embeddingsインスタンスを取得

    Titan Text Embeddings V2を使用（1536次元）
    """
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name=settings.aws_region
    )

    logger.info("Bedrock Embeddings initialized")
    return embeddings


def test_embeddings():
    """
    埋め込み生成のテスト
    """
    try:
        embeddings = get_embeddings()

        # テストテキスト
        test_text = "こんにちは、これはテストです。"

        logger.info(f"テキストを埋め込み化: {test_text}")

        # 埋め込みベクトルを生成
        vector = embeddings.embed_query(test_text)

        logger.info(f"埋め込みベクトルの次元数: {len(vector)}")
        logger.info(f"最初の5要素: {vector[:5]}")

        print("\n" + "="*50)
        print("✅ 埋め込み生成テスト成功！")
        print(f"ベクトル次元数: {len(vector)}")
        print(f"最初の5要素: {vector[:5]}")
        print("="*50 + "\n")

        return True

    except Exception as e:
        logger.error(f"埋め込み生成エラー: {e}", exc_info=True)
        print(f"\n❌ エラー: {e}\n")
        return False


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    test_embeddings()
