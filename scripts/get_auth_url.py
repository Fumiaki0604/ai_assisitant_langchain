"""
Google Drive認証URLを取得するスクリプト
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from google_auth_oauthlib.flow import InstalledAppFlow
from config.settings import settings

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

print("=" * 80)
print("Google Drive 認証URL取得")
print("=" * 80)

try:
    flow = InstalledAppFlow.from_client_secrets_file(
        settings.google_drive_credentials_path,
        SCOPES
    )

    print("\n認証を開始します...")
    print("WSL環境のため、以下の手順で認証してください:\n")
    print("1. これから表示されるURLをコピー")
    print("2. Windowsのブラウザで開く")
    print("3. Googleアカウントで認証")
    print("4. 認証後、ブラウザに表示されるURLをコピー")
    print("5. そのURLをここに貼り付ける\n")
    input("準備ができたらEnterキーを押してください...")

    # localhostサーバーを使わずに認証URLを生成
    auth_url, state = flow.authorization_url(
        access_type='offline',
        prompt='consent'
    )

    print("\n" + "=" * 80)
    print("以下のURLをブラウザにコピー&ペーストしてください:")
    print("=" * 80)
    print(f"\n{auth_url}\n")
    print("=" * 80)

    # リダイレクト後のURLを入力してもらう
    redirect_response = input("\n認証後、ブラウザのアドレスバーに表示されるURL全体を貼り付けてください: ").strip()

    if redirect_response:
        flow.fetch_token(authorization_response=redirect_response)
        creds = flow.credentials

        # トークンを保存
        import pickle
        os.makedirs(os.path.dirname(settings.google_drive_token_path), exist_ok=True)
        with open(settings.google_drive_token_path, 'wb') as token:
            pickle.dump(creds, token)

        print(f"\n✅ 認証成功! トークンを保存しました: {settings.google_drive_token_path}")

        # テスト
        from googleapiclient.discovery import build
        service = build('drive', 'v3', credentials=creds)
        results = service.files().list(pageSize=5, fields="files(id, name)").execute()
        files = results.get('files', [])

        print(f"\nアクセス可能なファイル（最初の5件）:")
        for f in files:
            print(f"  - {f['name']}")

except Exception as e:
    print(f"\n❌ エラー: {e}")
    import traceback
    traceback.print_exc()
