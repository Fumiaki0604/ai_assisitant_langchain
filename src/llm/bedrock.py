"""
AWS Bedrock (Claude 3.5 Sonnet) との連携
"""
import sys
import os

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from langchain_aws import ChatBedrock
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


def get_bedrock_llm() -> ChatBedrock:
    """
    AWS Bedrock LLMインスタンスを取得

    AWS認証は以下の順序で自動的に取得されます：
    1. 環境変数 (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    2. AWS CLI設定 (~/.aws/credentials)
    3. IAMロール（EC2/Lambda等で実行時）

    Returns:
        ChatBedrock: LangChainのBedrock LLMインスタンス
    """
    logger.info(f"Initializing Bedrock LLM: {settings.bedrock_model_id}")

    # AWS認証情報の設定（環境変数で指定されている場合）
    kwargs = {
        "model_id": settings.bedrock_model_id,
        "region_name": settings.aws_region,
        "model_kwargs": {
            "temperature": 0.7,
            "max_tokens": 2000,
        }
    }

    # 環境変数でAWSキーが明示的に指定されている場合のみ追加
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        logger.info("Using AWS credentials from environment variables")
        kwargs["credentials_profile_name"] = None  # 環境変数を優先
    else:
        logger.info("Using AWS credentials from AWS CLI configuration")

    llm = ChatBedrock(**kwargs)
    logger.info("Bedrock LLM initialized successfully")

    return llm


def test_bedrock_connection():
    """
    Bedrock接続テスト
    簡単なプロンプトを送信して接続を確認します
    """
    try:
        llm = get_bedrock_llm()
        response = llm.invoke("こんにちは。簡単に自己紹介してください。")
        logger.info(f"Bedrock connection test successful: {response.content[:100]}...")
        print("✅ Bedrock接続テスト成功！")
        print(f"応答: {response.content}")
        return True
    except Exception as e:
        logger.error(f"Bedrock connection test failed: {e}")
        print(f"❌ Bedrock接続テスト失敗: {e}")
        return False


if __name__ == "__main__":
    # このファイルを直接実行するとテストが実行されます
    logging.basicConfig(level=logging.INFO)
    test_bedrock_connection()
