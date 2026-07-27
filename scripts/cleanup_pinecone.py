"""
Pineconeの古いmercartデータ（source:file）を削除するスクリプト
"""
import sys
import io
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

from config.settings import settings
from pinecone import Pinecone
from src.rag.embeddings import get_embeddings

pc = Pinecone(api_key=settings.pinecone_api_key)
index = pc.Index(settings.pinecone_index_name)
embeddings = get_embeddings()

print("=== Pineconeクリーンアップ ===")
print(f"インデックス: {settings.pinecone_index_name}")

# source:fileかつmercartのベクターを検索
dummy_vec = embeddings.embed_query("メルカート カテゴリ 設定")

# mercartファイル由来のsource:fileベクターを取得
results = index.query(
    vector=dummy_vec,
    top_k=100,
    filter={"$and": [
        {"source": {"$eq": "file"}},
        {"file_path": {"$exists": True}}
    ]},
    include_metadata=True
)

mercart_ids = []
other_ids = []

for m in results['matches']:
    fp = m['metadata'].get('file_path', '')
    if 'mercart' in fp:
        mercart_ids.append(m['id'])
        print(f"[mercart] {m['id']} | {fp[:60]}")
    else:
        other_ids.append(m['id'])

print(f"\nmercart古いデータ: {len(mercart_ids)}件")
print(f"その他(削除しない): {len(other_ids)}件")

print(f"\n合計{len(mercart_ids)}件を削除します...")

# IDリストで削除（バッチ処理）
if mercart_ids:
    batch_size = 100
    for i in range(0, len(mercart_ids), batch_size):
        batch = mercart_ids[i:i+batch_size]
        index.delete(ids=batch)
        print(f"  削除: {i+len(batch)}/{len(mercart_ids)}件")
    print(f"削除完了: {len(mercart_ids)}件")
else:
    print("削除対象なし（すでにクリーンです）")

# フィルターベースで残りも削除（ページング対応）
print("\nフィルターベース削除でmercartの残存データをクリーンアップ...")
try:
    # mercartのfile_pathを持つsource:fileのベクターを全て削除
    # 複数回クエリして全件取得
    deleted_total = 0
    for _ in range(20):  # 最大20回ループ（安全策）
        res = index.query(
            vector=dummy_vec,
            top_k=1000,
            filter={"$and": [
                {"source": {"$eq": "file"}},
                {"file_path": {"$exists": True}}
            ]},
            include_metadata=True
        )
        remaining = [m['id'] for m in res['matches'] if 'mercart' in m['metadata'].get('file_path', '')]
        if not remaining:
            break
        index.delete(ids=remaining)
        deleted_total += len(remaining)
        print(f"  追加削除: {deleted_total}件")
    print(f"クリーンアップ完了。追加削除: {deleted_total}件")
except Exception as e:
    print(f"フィルター削除エラー（無視OK）: {e}")
