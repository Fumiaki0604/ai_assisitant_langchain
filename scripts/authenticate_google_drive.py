"""
Google Drive認証スクリプト（WSL対応）
"""
import sys
import os

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.auth.google_auth import GoogleDriveAuth
from config.settings import settings
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """
    Google Drive認証を実行
    """
    print("=" * 60)
    print("Google Drive 認証")
    print("=" * 60)

    print(f"\n認証情報ファイル: {settings.google_drive_credentials_path}")
    print(f"トークン保存先: {settings.google_drive_token_path}")

    if not os.path.exists(settings.google_drive_credentials_path):
        print(f"\n❌ エラー: 認証情報ファイルが見つかりません")
        print(f"   {settings.google_drive_credentials_path}")
        return

    auth = GoogleDriveAuth(
        settings.google_drive_credentials_path,
        settings.google_drive_token_path
    )

    try:
        print("\n認証を開始します...")
        print("\n⚠️  WSL環境では、ブラウザが自動的に開きません")
        print("    以下のURLを手動でブラウザにコピー&ペーストしてください:\n")

        # 認証を実行
        creds = auth.authenticate()

        if creds and creds.valid:
            print("\n✅ 認証成功!")
            print(f"トークンが保存されました: {settings.google_drive_token_path}")

            # テスト: ファイル一覧を取得
            service = auth.get_drive_service()
            results = service.files().list(
                pageSize=10,
                fields="files(id, name, mimeType)"
            ).execute()

            files = results.get('files', [])
            print(f"\nアクセス可能なファイル数: {len(files)}")

            if files:
                print("\n最初の数ファイル:")
                for file in files[:5]:
                    print(f"  - {file['name']}")
        else:
            print("\n❌ 認証に失敗しました")

    except KeyboardInterrupt:
        print("\n\n認証がキャンセルされました")
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        print(f"\n❌ エラー: {e}")


if __name__ == "__main__":
    main()
