"""
Google Drive OAuth 2.0 認証マネージャー
"""
import os
import sys
import pickle
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import logging

logger = logging.getLogger(__name__)

# Google Drive API のスコープ
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


class GoogleDriveAuth:
    """
    Google Drive 認証を管理するクラス
    """

    def __init__(self, credentials_path: str, token_path: str):
        """
        Args:
            credentials_path: OAuth 2.0 クライアント認証情報JSONファイルのパス
            token_path: 認証トークンを保存するパス
        """
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.creds = None

    def authenticate(self) -> Credentials:
        """
        Google Drive API の認証を実行

        Returns:
            Credentials: 認証済みのクレデンシャル
        """
        # トークンファイルが存在すれば読み込む
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                self.creds = pickle.load(token)
                logger.info("Loaded existing credentials from token file")

        # 認証が有効でない場合
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                # トークンをリフレッシュ
                logger.info("Refreshing expired credentials")
                self.creds.refresh(Request())
            else:
                # 新規認証フロー
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(
                        f"Credentials file not found: {self.credentials_path}\n"
                        "Please download OAuth 2.0 Client ID credentials from Google Cloud Console."
                    )

                logger.info("Starting new authentication flow")
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES
                )
                self.creds = flow.run_local_server(port=0)

            # トークンを保存
            os.makedirs(os.path.dirname(self.token_path), exist_ok=True)
            with open(self.token_path, 'wb') as token:
                pickle.dump(self.creds, token)
                logger.info(f"Saved credentials to {self.token_path}")

        return self.creds

    def get_drive_service(self):
        """
        認証済みの Google Drive API サービスを取得

        Returns:
            Resource: Google Drive API サービスオブジェクト
        """
        if not self.creds:
            self.authenticate()

        try:
            service = build('drive', 'v3', credentials=self.creds)
            logger.info("Successfully created Drive API service")
            return service
        except Exception as e:
            logger.error(f"Failed to create Drive service: {e}")
            raise


def get_google_drive_service(credentials_path: str, token_path: str):
    """
    Google Drive API サービスを取得するヘルパー関数

    Args:
        credentials_path: OAuth 2.0 クライアント認証情報JSONファイルのパス
        token_path: 認証トークンを保存するパス

    Returns:
        Resource: Google Drive API サービスオブジェクト
    """
    auth = GoogleDriveAuth(credentials_path, token_path)
    return auth.get_drive_service()


if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)

    # 設定ファイルから読み込む想定
    from config.settings import settings

    try:
        service = get_google_drive_service(
            settings.google_drive_credentials_path,
            settings.google_drive_token_path
        )

        # テスト: ファイル一覧を取得
        results = service.files().list(
            pageSize=10,
            fields="files(id, name, mimeType)"
        ).execute()

        files = results.get('files', [])
        print(f"\n認証成功！アクセス可能なファイル数: {len(files)}")

        if files:
            print("\n最初の数ファイル:")
            for file in files[:5]:
                print(f"  - {file['name']} ({file['mimeType']})")

    except Exception as e:
        print(f"\nエラー: {e}")
        sys.exit(1)
