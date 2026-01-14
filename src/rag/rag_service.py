"""
RAG（検索拡張生成）サービス
"""
import sys
import os

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.rag.embeddings import get_embeddings
from src.llm.bedrock import get_bedrock_llm
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


# RAG用のプロンプトテンプレート
RAG_PROMPT_TEMPLATE = """あなたは社内のAIアシスタントです。以下の参考情報を使用して、質問に正確に答えてください。

参考情報:
{context}

質問: {question}

回答の際は以下のルールに従ってください:
- 参考情報に基づいて、具体的かつ正確に答えてください
- 参考情報に答えがない場合は、「提供された情報からは回答できません。詳しくは担当部署にお問い合わせください。」と答えてください
- 簡潔で分かりやすい回答を心がけてください
- 必要に応じて箇条書きを使用してください

回答:"""


class RAGService:
    """
    RAG（検索拡張生成）サービスクラス
    """

    def __init__(self):
        """
        RAGサービスを初期化
        """
        # 環境変数を設定
        os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key

        # 埋め込みモデルを取得
        self.embeddings = get_embeddings()

        # Pineconeベクトルストアに接続
        logger.info(f"Connecting to Pinecone index: {settings.pinecone_index_name}")
        self.vectorstore = PineconeVectorStore(
            index_name=settings.pinecone_index_name,
            embedding=self.embeddings
        )

        # LLMを取得
        self.llm = get_bedrock_llm()

        # プロンプトテンプレート
        self.prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        # RetrievalQAチェーンを作成
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # 関連度の高い上位3件を取得
            ),
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )

        logger.info("RAG service initialized successfully")

    def answer_question(self, question: str) -> dict:
        """
        質問に対してRAGで回答を生成

        Args:
            question: 質問文

        Returns:
            dict: {
                "answer": 回答文,
                "sources": 参考にしたドキュメント情報のリスト
            }
        """
        try:
            logger.info(f"Processing question: {question[:50]}...")

            # RAG検索と回答生成
            result = self.qa_chain.invoke({"query": question})

            # 回答と参考ドキュメントを抽出
            answer = result['result']
            source_docs = result.get('source_documents', [])

            # 参考ドキュメント情報を整形
            sources = []
            for doc in source_docs:
                sources.append({
                    "title": doc.metadata.get('title', 'タイトルなし'),
                    "source": doc.metadata.get('source', '不明'),
                    "content": doc.page_content[:150]  # 最初の150文字
                })

            logger.info(f"Answer generated successfully with {len(sources)} source documents")

            return {
                "answer": answer,
                "sources": sources
            }

        except Exception as e:
            logger.error(f"Error in answer_question: {e}", exc_info=True)
            return {
                "answer": "申し訳ございません。回答の生成中にエラーが発生しました。もう一度お試しください。",
                "sources": []
            }


# シングルトンインスタンス
_rag_service_instance = None


def get_rag_service() -> RAGService:
    """
    RAGサービスのシングルトンインスタンスを取得

    Returns:
        RAGService: RAGサービスインスタンス
    """
    global _rag_service_instance

    if _rag_service_instance is None:
        _rag_service_instance = RAGService()

    return _rag_service_instance


if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)

    service = get_rag_service()

    test_question = "リモートワークは週に何日まで可能ですか？"
    print(f"\n質問: {test_question}\n")

    result = service.answer_question(test_question)

    print(f"回答: {result['answer']}\n")
    print(f"参考ドキュメント数: {len(result['sources'])}")
