"""
RAG検索と回答生成のテスト
"""
import sys
import os

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.rag.embeddings import get_embeddings
from src.llm.bedrock import get_bedrock_llm
from config.settings import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# RAG用のプロンプトテンプレート
RAG_PROMPT_TEMPLATE = """以下の情報を参考にして、質問に答えてください。
情報に答えがない場合は、「提供された情報からは回答できません」と答えてください。

参考情報:
{context}

質問: {question}

回答:"""


def test_rag_search():
    """
    RAG検索のテスト
    """
    try:
        # 環境変数を設定
        os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key

        # 埋め込みモデルを取得
        embeddings = get_embeddings()

        # Pineconeベクトルストアに接続
        logger.info(f"Pineconeインデックス '{settings.pinecone_index_name}' に接続中...")
        vectorstore = PineconeVectorStore(
            index_name=settings.pinecone_index_name,
            embedding=embeddings
        )

        # LLMを取得
        llm = get_bedrock_llm()

        # プロンプトテンプレート
        prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        # RetrievalQAチェーンを作成
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        # テスト質問
        test_questions = [
            "リモートワークは週に何日まで可能ですか？",
            "有給休暇の申請方法を教えてください",
            "タクシーはいつ使えますか？",
            "AWS Bedrockで利用できるモデルは何ですか？",
            "パスワードの要件を教えてください"
        ]

        print("\n" + "="*60)
        print("RAG検索テスト開始")
        print("="*60 + "\n")

        for i, question in enumerate(test_questions, 1):
            print(f"\n【質問{i}】: {question}")
            print("-" * 60)

            # RAG検索と回答生成
            result = qa_chain.invoke({"query": question})

            # 回答を表示
            print(f"\n【回答】:\n{result['result']}")

            # 参考にしたドキュメントを表示
            print(f"\n【参考ドキュメント】:")
            for j, doc in enumerate(result['source_documents'], 1):
                print(f"\n  [{j}] {doc.metadata.get('title', 'タイトルなし')}")
                print(f"  ソース: {doc.metadata.get('source', '不明')}")
                print(f"  内容: {doc.page_content[:100]}...")

            print("\n" + "="*60)

        print("\nRAG検索テスト完了")
        return True

    except Exception as e:
        logger.error(f"エラー: {e}", exc_info=True)
        print(f"\nエラー: {e}\n")
        return False


if __name__ == "__main__":
    test_rag_search()
