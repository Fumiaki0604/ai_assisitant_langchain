"""
RAGAS評価サービス
Bedrock LLM/Embeddingsを使用してRAGパイプラインを評価
"""
import sys
import os
import json
from datetime import datetime
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from src.llm.bedrock import get_bedrock_llm
from src.rag.embeddings import get_embeddings
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """RAGAS評価を実行するクラス"""

    def __init__(self):
        bedrock_llm = get_bedrock_llm()
        bedrock_embeddings = get_embeddings()

        self.ragas_llm = LangchainLLMWrapper(bedrock_llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(bedrock_embeddings)

        self.metrics = [
            Faithfulness(llm=self.ragas_llm),
            AnswerRelevancy(llm=self.ragas_llm, embeddings=self.ragas_embeddings),
            ContextPrecision(llm=self.ragas_llm),
            ContextRecall(llm=self.ragas_llm),
        ]

        logger.info("RAGAS evaluator initialized with Bedrock")

    def evaluate(self, evaluation_data: dict, batch_size: int = 10) -> dict:
        """
        RAGAS評価を実行

        Args:
            evaluation_data: {
                "question": List[str],
                "answer": List[str],
                "contexts": List[List[str]],
                "ground_truth": List[str]
            }
            batch_size: バッチサイズ

        Returns:
            評価結果dict
        """
        total = len(evaluation_data["question"])
        logger.info(f"Starting RAGAS evaluation: {total} cases")

        all_scores = []

        # バッチ処理
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            batch = {
                "question": evaluation_data["question"][i:end],
                "answer": evaluation_data["answer"][i:end],
                "contexts": evaluation_data["contexts"][i:end],
                "ground_truth": evaluation_data["ground_truth"][i:end],
            }

            logger.info(f"Evaluating batch {i // batch_size + 1} ({i+1}-{end}/{total})")

            try:
                dataset = Dataset.from_dict(batch)
                result = evaluate(
                    dataset=dataset,
                    metrics=self.metrics,
                )
                all_scores.append(result)
            except Exception as e:
                logger.error(f"Batch {i // batch_size + 1} failed: {e}")

        # 結果を集約
        return self._aggregate_results(all_scores, evaluation_data)

    def _aggregate_results(self, batch_results: list, evaluation_data: dict) -> dict:
        """バッチ結果を集約"""
        metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        aggregated = {name: [] for name in metric_names}

        for result in batch_results:
            df = result.to_pandas()
            for name in metric_names:
                if name in df.columns:
                    aggregated[name].extend(df[name].tolist())

        # 統計計算
        summary = {}
        for name, scores in aggregated.items():
            valid_scores = [s for s in scores if s is not None and s == s]  # NaN除外
            if valid_scores:
                summary[name] = {
                    "mean": round(sum(valid_scores) / len(valid_scores), 4),
                    "min": round(min(valid_scores), 4),
                    "max": round(max(valid_scores), 4),
                    "count": len(valid_scores),
                }
            else:
                summary[name] = {"mean": 0, "min": 0, "max": 0, "count": 0}

        # 個別スコア
        per_case = []
        for i in range(len(evaluation_data["question"])):
            case = {
                "question": evaluation_data["question"][i],
                "ground_truth": evaluation_data["ground_truth"][i],
                "answer": evaluation_data["answer"][i],
            }
            for name in metric_names:
                if i < len(aggregated[name]):
                    case[name] = aggregated[name][i]
            per_case.append(case)

        return {
            "timestamp": datetime.now().isoformat(),
            "total_cases": len(evaluation_data["question"]),
            "summary": summary,
            "per_case": per_case,
        }

    def save_results(self, results: dict, output_dir: str = None) -> str:
        """結果をJSON+Markdownで保存"""
        output_dir = output_dir or settings.evaluation_results_path
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON保存
        json_path = os.path.join(output_dir, f"eval_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Markdownレポート生成
        md_path = os.path.join(output_dir, f"eval_{timestamp}.md")
        report = self._generate_report(results)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Results saved: {json_path}")
        return json_path

    def _generate_report(self, results: dict) -> str:
        """Markdownレポートを生成"""
        lines = [
            f"# RAG評価レポート",
            f"- 日時: {results['timestamp']}",
            f"- テストケース数: {results['total_cases']}",
            "",
            "## 総合スコア",
            "| メトリクス | 平均 | 最小 | 最大 |",
            "|-----------|------|------|------|",
        ]

        metric_labels = {
            "faithfulness": "Faithfulness（忠実性）",
            "answer_relevancy": "Answer Relevancy（回答関連性）",
            "context_precision": "Context Precision（検索精度）",
            "context_recall": "Context Recall（検索網羅性）",
        }

        for name, label in metric_labels.items():
            s = results["summary"].get(name, {})
            lines.append(f"| {label} | {s.get('mean', 0):.2%} | {s.get('min', 0):.2%} | {s.get('max', 0):.2%} |")

        # 低スコアケースを抽出
        low_cases = [
            c for c in results.get("per_case", [])
            if any(c.get(m, 1.0) < 0.5 for m in metric_labels)
        ]

        if low_cases:
            lines.extend(["", "## 要改善ケース（スコア < 0.5）"])
            for i, case in enumerate(low_cases, 1):
                lines.append(f"\n### {i}. {case['question'][:80]}")
                lines.append(f"- 期待回答: {case['ground_truth'][:100]}")
                lines.append(f"- 実際回答: {case['answer'][:100]}")
                for name, label in metric_labels.items():
                    score = case.get(name)
                    if score is not None and score < 0.5:
                        lines.append(f"- **{label}: {score:.2%}**")

        return "\n".join(lines)
