import sys, io, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from dotenv import load_dotenv
load_dotenv()
from config.settings import settings
from pinecone import Pinecone
from src.rag.embeddings import get_embeddings

pc = Pinecone(api_key=settings.pinecone_api_key)
index = pc.Index(settings.pinecone_index_name)

stats = index.describe_index_stats()
print(f"総ベクター数: {stats['total_vector_count']}")

embeddings = get_embeddings()
vec = embeddings.embed_query('メルカートのカテゴリ設定方法')
res = index.query(vector=vec, top_k=3, filter={"source": {"$eq": "mercart"}}, include_metadata=True)
print(f"source:mercartのヒット数: {len(res['matches'])}")
for m in res['matches']:
    print(f"  {m['metadata'].get('category','?')} | {m['metadata'].get('subcategory','?')}")
