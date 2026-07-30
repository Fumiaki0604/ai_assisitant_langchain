"""ドキュメントリランキング・グラウンディング検証"""
import re
import logging
from typing import List
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

HF_RERANKER_MODEL = "hotchpotch/japanese-reranker-xsmall-v2"


class Reranker:
    def __init__(self, llm, cohere_api_key: str = None):
        self.llm = llm
        self.cohere_client = None
        self.hf_model = None

        # HuggingFace日本語リランカーを優先で使用
        if HF_AVAILABLE:
            try:
                self.hf_model = CrossEncoder(HF_RERANKER_MODEL, max_length=512)
                logger.info(f"HuggingFace Reranker enabled: {HF_RERANKER_MODEL}")
            except Exception as e:
                logger.warning(f"HuggingFace Reranker load failed: {e}")

        # HFが使えない場合はCohereにフォールバック
        if self.hf_model is None and COHERE_AVAILABLE and cohere_api_key:
            self.cohere_client = cohere.Client(cohere_api_key)
            logger.info("Cohere Rerank enabled (fallback)")

    def _keyword_score(self, question: str, doc_content: str) -> float:
        question_words = set(re.findall(r'\w+', question.lower()))
        doc_words = set(re.findall(r'\w+', doc_content.lower()))
        if not question_words:
            return 0.0
        return len(question_words & doc_words) / len(question_words)

    def rerank_documents(self, question: str, docs: List[Document], top_n: int = 3, min_score: float = 0.5) -> tuple:
        """ドキュメントをリランキングし関連度スコアでフィルタリング

        Returns:
            tuple: (filtered_docs, top_score)
        """
        if not docs:
            return [], 0.0

        # HuggingFace日本語リランカー
        if self.hf_model:
            try:
                texts = [doc.page_content for doc in docs]
                pairs = [(question, t) for t in texts]
                scores = self.hf_model.predict(pairs).tolist()
                scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
                top_score = scored[0][0] if scored else 0.0
                # xsmall-v2のスコアはlogit値（0〜1に正規化不要、相対比較で使う）
                filtered = [doc for score, doc in scored[:top_n]]
                logger.info(f"HF reranked {len(docs)} -> {len(filtered)} docs (top_score={top_score:.3f})")
                return filtered, min(top_score, 1.0)
            except Exception as e:
                logger.warning(f"HF rerank failed, using fallback: {e}")

        # CohereフォールバックはHFが使えない場合のみ
        if self.cohere_client:
            try:
                texts = [doc.page_content for doc in docs]
                response = self.cohere_client.rerank(
                    model="rerank-multilingual-v3.0",
                    query=question,
                    documents=texts,
                    top_n=top_n
                )
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

        # フォールバック: ハイブリッドスコア（順位 + キーワード）
        scored_docs = []
        for i, doc in enumerate(docs):
            vector_score = 1.0 / (i + 1)
            keyword_score = self._keyword_score(question, doc.page_content)
            hybrid_score = 0.7 * vector_score + 0.3 * keyword_score
            scored_docs.append((hybrid_score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_score = scored_docs[0][0] if scored_docs else 0.0
        filtered = [(s, d) for s, d in scored_docs[:top_n] if s >= min_score]
        return [doc for _, doc in filtered], top_score

    def verify_grounding(self, answer: str, sources: List[Document]) -> dict:
        """回答がソースに基づいているか検証"""
        if not sources or not answer:
            return {"is_grounded": False, "grounding_score": 0.0, "warning": "参考情報なし"}

        try:
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

            is_grounded = "grounded" in result.lower() and "ungrounded" not in result.lower()
            is_partial = "partial" in result.lower()

            if is_grounded:
                grounding_score, warning = 1.0, None
            elif is_partial:
                grounding_score = 0.5
                warning = "一部の情報は参考資料に基づいていない可能性があります"
            else:
                grounding_score = 0.0
                warning = "回答が参考資料に基づいていない可能性があります"

            logger.info(f"Answer grounding: {grounding_score:.1f}")
            return {"is_grounded": is_grounded or is_partial, "grounding_score": grounding_score, "warning": warning}

        except Exception as e:
            logger.warning(f"Answer grounding check failed: {e}")
            return {"is_grounded": True, "grounding_score": 0.5, "warning": None}
