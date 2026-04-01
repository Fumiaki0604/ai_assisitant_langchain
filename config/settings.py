"""
アプリケーション設定管理
環境変数から設定を読み込みます
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """アプリケーション設定"""

    # AWS Bedrock
    aws_region: str = "us-west-2"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    bedrock_model_id: str = "us.anthropic.claude-sonnet-4-6"

    # Pinecone
    pinecone_api_key: str
    pinecone_environment: str
    pinecone_index_name: str = "slack-ai-assistant"

    # Slack
    slack_bot_token: str
    slack_app_token: str
    slack_signing_secret: str
    slack_auto_reply_channels: str = ""  # カンマ区切りのチャンネルID（自動返信対象）
    slack_knowledge_channels: str = ""  # カンマ区切りのチャンネルID（RAG取り込みのみ、返信なし）

    # Confluence (Optional)
    confluence_base_url: Optional[str] = None
    confluence_api_token: Optional[str] = None
    confluence_user_email: Optional[str] = None

    # Notion (Optional)
    notion_api_key: Optional[str] = None

    # Google Drive (Optional)
    google_drive_credentials_path: str = "credentials/google_credentials.json"
    google_drive_token_path: str = "credentials/google_token.json"
    google_drive_folder_id: Optional[str] = None

    # Cohere (Optional - for Reranking)
    cohere_api_key: Optional[str] = None

    # S3 (差分インデックス状態管理)
    s3_state_bucket: Optional[str] = None

    # Pinecone Hybrid Search
    pinecone_hybrid_alpha: float = 0.7  # 0=sparse only, 1=dense only

    # Application
    log_level: str = "INFO"
    feedback_log_file: str = "logs/feedback.json"

    # Evaluation
    evaluation_dataset_path: str = "data/evaluation/testset.jsonl"
    evaluation_results_path: str = "data/evaluation/results"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# グローバル設定インスタンス
settings = Settings()
