"""
RAG（検索拡張生成）サービス
"""
import sys
import os
import re
import requests
from typing import List

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from src.rag.embeddings import get_embeddings
from src.llm.bedrock import get_bedrock_llm
from config.settings import settings
import logging

# Cohere for Reranking
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

logger = logging.getLogger(__name__)

# 回答不可を示すフレーズ
UNABLE_TO_ANSWER_PHRASES = [
    "提供された情報からは回答できません",
    "回答できません",
    "情報がありません",
    "見つかりませんでした",
]


# RAG用のプロンプトテンプレート
RAG_PROMPT_TEMPLATE = """あなたは社内のAIアシスタントです。参考情報のみを根拠として回答してください。

## 参考情報
{context}

## 質問
{question}

## 回答ルール
1. 参考情報に記載された内容のみを使用する（推測・一般知識は使わない）
2. 回答の根拠となる情報源を明示する（例：「〇〇によると...」）
3. 部分的にしか情報がない場合は、分かる範囲で回答し、不明点を明示する
4. 情報がない場合は「提供された情報からは回答できません。担当部署にお問い合わせください。」と答える
5. 箇条書きで構造化し、要点を先に述べる

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

        # Retrieverを作成（候補を多めに取得してリランキング）
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 10}
        )

        # Cohere Rerankクライアント
        self.cohere_client = None
        if COHERE_AVAILABLE and settings.cohere_api_key:
            self.cohere_client = cohere.Client(settings.cohere_api_key)
            logger.info("Cohere Rerank enabled")

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

    def _keyword_score(self, question: str, doc_content: str) -> float:
        """キーワードマッチングスコアを計算（ハイブリッド検索用）"""
        question_words = set(re.findall(r'\w+', question.lower()))
        doc_words = set(re.findall(r'\w+', doc_content.lower()))
        if not question_words:
            return 0.0
        matched = len(question_words & doc_words)
        return matched / len(question_words)

    def _rerank_documents(self, question: str, docs: List[Document], top_n: int = 3, min_score: float = 0.3) -> tuple:
        """ドキュメントをリランキングし、関連度スコアでフィルタリング

        Returns:
            tuple: (filtered_docs, top_score) - フィルタ済みドキュメントと最高スコア
        """
        if not docs:
            return [], 0.0

        # Cohereでリランキング
        if self.cohere_client:
            try:
                texts = [doc.page_content for doc in docs]
                response = self.cohere_client.rerank(
                    model="rerank-multilingual-v3.0",
                    query=question,
                    documents=texts,
                    top_n=top_n
                )

                # スコアでフィルタリング
                filtered = []
                top_score = 0.0
                for r in response.results:
                    if r.relevance_score >= min_score:
                        filtered.append(docs[r.index])
                    if r.relevance_score > top_score:
                        top_score = r.relevance_score

                logger.info(f"Cohere reranked {len(docs)} -> {len(filtered)} docs (top_score={top_score:.2f}, min={min_score})")
                return filtered, top_score
            except Exception as e:
                logger.warning(f"Cohere rerank failed, using hybrid: {e}")

        # フォールバック: ハイブリッドスコア（ベクトル順位 + キーワード）
        scored_docs = []
        for i, doc in enumerate(docs):
            vector_score = 1.0 / (i + 1)  # 順位スコア
            keyword_score = self._keyword_score(question, doc.page_content)
            hybrid_score = 0.7 * vector_score + 0.3 * keyword_score
            scored_docs.append((hybrid_score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_score = scored_docs[0][0] if scored_docs else 0.0
        filtered = [(s, d) for s, d in scored_docs[:top_n] if s >= min_score]
        return [doc for _, doc in filtered], top_score

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

    def _classify_question_type(self, question: str) -> str:
        """質問が内部製品に関するものか外部サービスに関するものかをLLMで判定"""
        try:
            prompt = """以下の質問が「自社製品(ecbeing/メルカート/visumo等)に関する技術的な質問」か「外部サービス/一般的な技術の質問」かを判定してください。

質問: {question}

判定基準:
- 自社製品の機能、設定、運用、事例に関する質問 → internal
- Google、Bing、AWS等の外部サービスに関する質問 → external
- SEO、インデックス、サーバー等の一般技術で自社製品に限定されない質問 → external

回答は「internal」か「external」の1単語のみ:"""

            result = self.llm.invoke(prompt.format(question=question))
            if hasattr(result, 'content'):
                result = result.content

            classification = "external" if "external" in result.lower() else "internal"
            logger.info(f"Question classified as: {classification}")
            return classification
        except Exception as e:
            logger.warning(f"Question classification failed: {e}")
            return "internal"

    def _verify_answer_grounding(self, answer: str, sources: List[Document]) -> dict:
        """回答がソースに基づいているか検証"""
        if not sources or not answer:
            return {"is_grounded": False, "grounding_score": 0.0, "warning": "参考情報なし"}

        try:
            # ソースの内容を結合
            source_text = "\n".join([doc.page_content[:500] for doc in sources[:3]])

            prompt = """以下の「回答」が「参考情報」に基づいているか検証してください。

## 参考情報
{sources}

## 回答
{answer}

## 検証基準
- 回答の主張が参考情報に記載されている → grounded
- 回答が参考情報にない情報を含む（推測・一般知識）→ ungrounded
- 部分的に基づいている → partial

検証結果を以下の形式で回答:
結果: [grounded/partial/ungrounded]
理由: [1行で簡潔に]"""

            result = self.llm.invoke(prompt.format(sources=source_text, answer=answer[:500]))
            if hasattr(result, 'content'):
                result = result.content

            # 結果をパース
            is_grounded = "grounded" in result.lower() and "ungrounded" not in result.lower()
            is_partial = "partial" in result.lower()

            if is_grounded:
                grounding_score = 1.0
                warning = None
            elif is_partial:
                grounding_score = 0.5
                warning = "一部の情報は参考資料に基づいていない可能性があります"
            else:
                grounding_score = 0.0
                warning = "回答が参考資料に基づいていない可能性があります"

            logger.info(f"Answer grounding: {grounding_score:.1f} ({result[:50]}...)")
            return {"is_grounded": is_grounded or is_partial, "grounding_score": grounding_score, "warning": warning}

        except Exception as e:
            logger.warning(f"Answer grounding check failed: {e}")
            return {"is_grounded": True, "grounding_score": 0.5, "warning": None}

    def web_search(self, query: str, num_results: int = 3) -> list:
        """DuckDuckGoでWeb検索を実行"""
        try:
            # DuckDuckGo HTML検索
            headers = {'User-Agent': 'Mozilla/5.0'}
            params = {'q': query, 'kl': 'jp-jp'}
            response = requests.get(
                'https://html.duckduckgo.com/html/',
                params=params,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()

            # 結果をパース
            results = []
            # DuckDuckGoの結果リンクを抽出
            pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>'
            matches = re.findall(pattern, response.text)

            for url, title in matches[:num_results]:
                # URLデコード
                if url.startswith('//duckduckgo.com/l/?uddg='):
                    url = requests.utils.unquote(url.split('uddg=')[1].split('&')[0])
                results.append({'url': url, 'title': title.strip()})

            logger.info(f"Web search found {len(results)} results for: {query[:30]}...")
            return results
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

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

            # 関連ドキュメントを検索（k=10で取得）
            candidate_docs = self.retriever.invoke(question)

            # リランキングで上位3件を選択（関連度スコア0.3未満は除外）
            source_docs, top_score = self._rerank_documents(question, candidate_docs, top_n=3, min_score=0.3)

            # ソースを分類
            sources_by_type = self._classify_sources(source_docs)

            # Web検索の判定
            web_results = []
            should_web_search = False

            # 内部ソースがない、または関連度が低い場合
            if not sources_by_type['drive'] and not sources_by_type['slack']:
                should_web_search = True
                logger.info("No internal sources found")
            elif top_score < 0.5:
                # 関連度が中程度以下の場合、質問タイプで判断
                question_type = self._classify_question_type(question)
                if question_type == "external":
                    should_web_search = True
                    logger.info(f"Low relevance ({top_score:.2f}) + external question -> web search")

            if should_web_search:
                web_results = self.web_search(question)
                sources_by_type['web'] = web_results

            # コンテキストを構築
            context_parts = [doc.page_content for doc in source_docs]
            if url_content:
                context_parts.append(f"URLの内容:\n{url_content}")
            # Web検索結果もコンテキストに追加
            if web_results:
                web_context = "Web検索結果:\n" + "\n".join([f"- {r['title']}: {r['url']}" for r in web_results])
                context_parts.append(web_context)
            context = "\n\n".join(context_parts)

            # プロンプトを生成
            formatted_prompt = self.prompt.format(context=context, question=question)

            # LLMで回答を生成
            answer = self.llm.invoke(formatted_prompt)
            if hasattr(answer, 'content'):
                answer = answer.content

            # 回答不可判定
            is_unable = self._is_unable_to_answer(answer)

            # 回答の整合性検証
            grounding_result = {"is_grounded": True, "grounding_score": 1.0, "warning": None}
            if not is_unable and source_docs:
                grounding_result = self._verify_answer_grounding(answer, source_docs)

            # 信頼度スコアを計算（関連度 × 整合性）
            confidence_score = top_score * grounding_result["grounding_score"]

            logger.info(f"Answer generated (unable={is_unable}, confidence={confidence_score:.2f}), sources: slack={len(sources_by_type['slack'])}, drive={len(sources_by_type['drive'])}, web={len(sources_by_type.get('web', []))}")

            return {
                "answer": answer,
                "sources_by_type": sources_by_type,
                "is_unable_to_answer": is_unable,
                "confidence_score": confidence_score,
                "relevance_score": top_score,
                "grounding_warning": grounding_result.get("warning")
            }

        except Exception as e:
            logger.error(f"Error in answer_question: {e}", exc_info=True)
            return {
                "answer": "申し訳ございません。回答の生成中にエラーが発生しました。もう一度お試しください。",
                "sources_by_type": {"slack": [], "drive": [], "other": [], "web": []},
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
    print(f"参考ドキュメント: slack={len(result['sources_by_type']['slack'])}, drive={len(result['sources_by_type']['drive'])}")
