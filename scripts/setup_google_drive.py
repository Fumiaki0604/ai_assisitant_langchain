"""
Google Driveのドキュメントを読み込んでPineconeに登録するスクリプト
"""
import sys
import os

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.loaders.google_drive_loader import load_documents_from_google_drive
from config.settings import settings
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """
    Google DriveからドキュメントをPineconeに登録
    """
    print("=" * 60)
    print("Google Drive → Pinecone セットアップ")
    print("=" * 60)

    # 設定確認
    folder_id = settings.google_drive_folder_id
    if folder_id:
        print(f"\n対象フォルダID: {folder_id}")
    else:
        print("\n対象: アクセス可能な全てのファイル")
        print("（特定のフォルダのみ対象にする場合は .env で GOOGLE_DRIVE_FOLDER_ID を設定してください）")

    print("\n認証情報:")
    print(f"  - Credentials: {settings.google_drive_credentials_path}")
    print(f"  - Token: {settings.google_drive_token_path}")

    # 認証情報ファイルの存在確認
    if not os.path.exists(settings.google_drive_credentials_path):
        print(f"\n❌ エラー: 認証情報ファイルが見つかりません")
        print(f"   パス: {settings.google_drive_credentials_path}")
        print("\n【セットアップ手順】")
        print("1. Google Cloud Console (https://console.cloud.google.com) にアクセス")
        print("2. プロジェクトを作成 or 選択")
        print("3. Google Drive API を有効化")
        print("4. 「認証情報」→「認証情報を作成」→「OAuth クライアント ID」")
        print("5. アプリケーションの種類: デスクトップアプリ")
        print("6. JSONをダウンロードして上記パスに配置")
        return

    print("\n" + "=" * 60)
    input("Enterキーを押すと処理を開始します（初回は認証画面が開きます）...")
    print()

    try:
        # ドキュメントを読み込んでPineconeに保存
        success, count = load_documents_from_google_drive(folder_id)

        print("\n" + "=" * 60)
        if success:
            print(f"✅ 完了: {count} 件のドキュメントをPineconeに登録しました")
        else:
            print("❌ エラー: ドキュメントの登録に失敗しました")
            print("ログを確認してください")
        print("=" * 60)

    except Exception as e:
        logger.error(f"処理中にエラーが発生: {e}", exc_info=True)
        print(f"\n❌ エラー: {e}")


if __name__ == "__main__":
    main()
