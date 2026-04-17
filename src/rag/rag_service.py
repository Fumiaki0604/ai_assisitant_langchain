"""
RAG（検索拡張生成）サービス
"""
import sys
import os
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from src.rag.embeddings import get_embeddings
from src.rag.prompts import (
    RAG_PROMPT_TEMPLATE, INTERNAL_SEARCH_PROMPT_TEMPLATE,
    WEB_FALLBACK_PROMPT_TEMPLATE, UNABLE_TO_ANSWER_PHRASES,
)
from src.rag.classifier import Classifier
from src.rag.reranker import Reranker
from src.rag.web_searcher import fetch_url_content, web_search
from src.llm.bedrock import get_bedrock_llm
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key

        self.embeddings = get_embeddings()

        # BM25エンコーダー（Hybrid Search用）
        from pinecone_text.sparse import BM25Encoder
        try:
            bm25_local = "/app/data/bm25_params.json"
            if os.path.exists(bm25_local):
                self.bm25 = BM25Encoder()
                self.bm25.load(bm25_local)
                logger.info("BM25Encoder loaded from local file")
            else:
                self.bm25 = BM25Encoder().default()
                logger.info("BM25Encoder initialized from remote (default)")
        except Exception as e:
            logger.error(f"BM25Encoder init failed: {e}", exc_info=True)
            raise

        logger.info(f"Connecting to Pinecone index: {settings.pinecone_index_name}")

        self.llm = get_bedrock_llm()
        self.classifier = Classifier(self.llm)
        self.reranker = Reranker(self.llm, settings.cohere_api_key)

        self.prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        self.web_fallback_prompt = PromptTemplate(
            template=WEB_FALLBACK_PROMPT_TEMPLATE,
            input_variables=["web_context", "internal_context", "question"]
        )

        logger.info("RAG service initialized successfully")

    def classify_message_intent(self, message: str) -> dict:
        return self.classifier.classify_message_intent(message)

    def fetch_url_content(self, url: str) -> str:
        return fetch_url_content(url)

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
                "link": doc.metadata.get('web_view_link') or doc.metadata.get('permalink', ''),
                "content": doc.page_content[:200]
            }

            source_type = doc.metadata.get('source', '').lower()
            if source_type == 'slack':
                slack_sources.append(source_info)
            elif source_type == 'google_drive':
                drive_sources.append(source_info)
            else:
                other_sources.append(source_info)

        return {"slack": slack_sources, "drive": drive_sources, "other": other_sources}

    def _hybrid_search(self, question: str, top_k: int = 10) -> List[Document]:
        """Hybrid Search（dense Titan + sparse BM25）でPineconeを検索"""
        from pinecone import Pinecone
        alpha = settings.pinecone_hybrid_alpha

        dense = self.embeddings.embed_query(question)
        sparse = self.bm25.encode_queries(question)

        scaled_dense = [v * alpha for v in dense]
        scaled_sparse = {
            "indices": sparse["indices"],
            "values": [v * (1 - alpha) for v in sparse["values"]],
        }

        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index_name)
        results = index.query(
            vector=scaled_dense,
            sparse_vector=scaled_sparse,
            top_k=top_k,
            include_metadata=True,
        )

        docs = []
        for match in results.matches:
            metadata = dict(match.metadata)
            text = metadata.pop("text", "")
            docs.append(Document(page_content=text, metadata=metadata))
        logger.info(f"Hybrid search returned {len(docs)} docs for: {question[:30]}...")
        return docs

    def _is_unable_to_answer(self, answer: str) -> bool:
        return any(phrase in answer for phrase in UNABLE_TO_ANSWER_PHRASES)

    def _invoke_with_images(self, text_prompt: str, images: list) -> str:
        """画像付きプロンプトでLLMを呼び出し"""
        content_blocks = []
        for img in images:
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img["media_type"],
                    "data": img["base64_data"],
                }
            })
        content_blocks.append({"type": "text", "text": text_prompt})
        result = self.llm.invoke([HumanMessage(content=content_blocks)])
        return result.content if hasattr(result, 'content') else result

    def answer_question(self, question: str, url_content: str = "", images: list = None, skip_web_search: bool = False) -> dict:
        """質問に対してRAGで回答を生成"""
        try:
            logger.info(f"Processing question: {question[:50]}...")

            candidate_docs = self._hybrid_search(question)
            source_docs, top_score = self.reranker.rerank_documents(question, candidate_docs, top_n=3, min_score=0.5)
            sources_by_type = self._classify_sources(source_docs)

            request_type = self.classifier.classify_request_type(question)
            is_internal_only = request_type in ("experience", "document", "owner")
            if is_internal_only:
                logger.info(f"Request type '{request_type}' detected -> skip web search, use INTERNAL_SEARCH prompt")

            # Web検索判定
            web_results = []
            should_web_search = False
            if not is_internal_only and not skip_web_search:
                has_internal_sources = sources_by_type['drive'] or sources_by_type['slack']
                if not has_internal_sources:
                    should_web_search = True
                    logger.info("No internal sources found -> web search")
                elif top_score < 0.5:
                    should_web_search = True
                    logger.info(f"Low relevance ({top_score:.2f}) -> web search")

            if should_web_search:
                web_results = web_search(question)
                sources_by_type['web'] = web_results

            # プロンプト選択とコンテキスト構築
            if is_internal_only:
                context_parts = [doc.page_content for doc in source_docs]
                context = "\n\n".join(context_parts) if context_parts else "該当する社内情報なし"
                internal_search_prompt = PromptTemplate(
                    template=INTERNAL_SEARCH_PROMPT_TEMPLATE,
                    input_variables=["context", "question"]
                )
                formatted_prompt = internal_search_prompt.format(context=context, question=question)
            elif should_web_search:
                web_context_parts = []
                for r in web_results:
                    entry = f"### {r['title']}\nURL: {r['url']}"
                    if r.get('content'):
                        entry += f"\n{r['content']}"
                    web_context_parts.append(entry)
                web_context = "\n\n".join(web_context_parts)

                internal_context_parts = [doc.page_content for doc in source_docs]
                if url_content:
                    internal_context_parts.append(f"URLの内容:\n{url_content}")
                internal_context = "\n\n".join(internal_context_parts) if internal_context_parts else "該当する社内情報なし"

                formatted_prompt = self.web_fallback_prompt.format(
                    web_context=web_context,
                    internal_context=internal_context,
                    question=question
                )
            else:
                context_parts = [doc.page_content for doc in source_docs]
                if url_content:
                    context_parts.append(f"URLの内容:\n{url_content}")
                context = "\n\n".join(context_parts)
                formatted_prompt = self.prompt.format(context=context, question=question)

            # LLMで回答生成
            if images:
                answer = self._invoke_with_images(formatted_prompt, images)
            else:
                answer = self.llm.invoke(formatted_prompt)
                if hasattr(answer, 'content'):
                    answer = answer.content

            is_unable = self._is_unable_to_answer(answer)

            grounding_result = {"is_grounded": True, "grounding_score": 1.0, "warning": None}
            if not should_web_search and not is_unable and source_docs:
                grounding_result = self.reranker.verify_grounding(answer, source_docs)

            if should_web_search:
                confidence_score = 0.7
            else:
                confidence_score = top_score * grounding_result["grounding_score"]

            logger.info(
                f"Answer generated (unable={is_unable}, confidence={confidence_score:.2f}), "
                f"sources: slack={len(sources_by_type['slack'])}, drive={len(sources_by_type['drive'])}, "
                f"web={len(sources_by_type.get('web', []))}"
            )

            return {
                "answer": answer,
                "sources_by_type": sources_by_type,
                "is_unable_to_answer": is_unable,
                "confidence_score": confidence_score,
                "relevance_score": top_score,
                "grounding_warning": grounding_result.get("warning"),
                "question_type": request_type,
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
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    service = get_rag_service()
    result = service.answer_question("リモートワークは週に何日まで可能ですか？")
    print(f"回答: {result['answer']}")
