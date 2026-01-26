"""
RAG（検索拡張生成）サービス
"""
import sys
import os
import re
import requests

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.rag.embeddings import get_embeddings
from src.llm.bedrock import get_bedrock_llm
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

# 回答不可を示すフレーズ
UNABLE_TO_ANSWER_PHRASES = [
    "提供された情報からは回答できません",
    "回答できません",
    "情報がありません",
    "見つかりませんでした",
]


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

        # Retrieverを作成
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}  # 関連度の高い上位3件を取得
        )

        logger.info("RAG service initialized successfully")

    def _classify_sources(self, source_docs) -> dict:
        """ソースをカテゴリ別に分類し重複排除"""
        slack_sources = []
        drive_sources = []
        other_sources = []
        seen_titles = set()

        for doc in source_docs:
            title = doc.metadata.get('title', 'タイトルなし')
            if title in seen_titles:
                continue
            seen_titles.add(title)

            source_info = {
                "title": title,
                "source": doc.metadata.get('source', '不明'),
                "link": doc.metadata.get('web_view_link', ''),
                "content": doc.page_content[:200]
            }

            source_type = doc.metadata.get('source', '').lower()
            if source_type == 'slack':
                slack_sources.append(source_info)
            elif source_type == 'google_drive':
                drive_sources.append(source_info)
            else:
                other_sources.append(source_info)

        return {
            "slack": slack_sources,
            "drive": drive_sources,
            "other": other_sources
        }

    def _is_unable_to_answer(self, answer: str) -> bool:
        """回答不可かどうかを判定"""
        return any(phrase in answer for phrase in UNABLE_TO_ANSWER_PHRASES)

    def fetch_url_content(self, url: str) -> str:
        """URLの内容を取得"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            # HTMLタグを除去して最初の2000文字を返す
            text = re.sub(r'<[^>]+>', '', response.text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:2000]
        except Exception as e:
            logger.error(f"Failed to fetch URL {url}: {e}")
            return ""

    def answer_question(self, question: str, url_content: str = "") -> dict:
        """
        質問に対してRAGで回答を生成

        Args:
            question: 質問文
            url_content: URLから取得したコンテンツ（オプション）

        Returns:
            dict: {
                "answer": 回答文,
                "sources_by_type": ソース別の参考ドキュメント,
                "is_unable_to_answer": 回答不可フラグ
            }
        """
        try:
            logger.info(f"Processing question: {question[:50]}...")

            # 関連ドキュメントを検索
            source_docs = self.retriever.invoke(question)

            # コンテキストを構築
            context_parts = [doc.page_content for doc in source_docs]
            if url_content:
                context_parts.append(f"URLの内容:\n{url_content}")
            context = "\n\n".join(context_parts)

            # プロンプトを生成
            formatted_prompt = self.prompt.format(context=context, question=question)

            # LLMで回答を生成
            answer = self.llm.invoke(formatted_prompt)
            if hasattr(answer, 'content'):
                answer = answer.content

            # ソースを分類
            sources_by_type = self._classify_sources(source_docs)

            # 回答不可判定
            is_unable = self._is_unable_to_answer(answer)

            logger.info(f"Answer generated (unable={is_unable}), sources: slack={len(sources_by_type['slack'])}, drive={len(sources_by_type['drive'])}")

            return {
                "answer": answer,
                "sources_by_type": sources_by_type,
                "is_unable_to_answer": is_unable
            }

        except Exception as e:
            logger.error(f"Error in answer_question: {e}", exc_info=True)
            return {
                "answer": "申し訳ございません。回答の生成中にエラーが発生しました。もう一度お試しください。",
                "sources_by_type": {"slack": [], "drive": [], "other": []},
                "is_unable_to_answer": True
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
