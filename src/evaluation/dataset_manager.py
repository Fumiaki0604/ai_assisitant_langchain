"""
評価データセット管理
テストケースの読み込み・フィードバックからの抽出
"""
import sys
import os
import json
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rag.rag_service import get_rag_service
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class DatasetManager:
    """評価データセットの管理"""

    def load_testset(self, path: str = None) -> List[dict]:
        """
        テストセット（JSONL）を読み込み

        各行: {"question": str, "ground_truth": str}
        """
        path = path or settings.evaluation_dataset_path
        if not os.path.exists(path):
            logger.error(f"Testset not found: {path}")
            return []

        cases = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))

        logger.info(f"Loaded {len(cases)} test cases from {path}")
        return cases

    def load_negative_feedbacks(self, limit: int = 20) -> List[dict]:
        """フィードバックログからネガティブケースを抽出"""
        log_file = settings.feedback_log_file
        if not os.path.exists(log_file):
            return []

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

        negatives = [
            {"question": log["question"], "ground_truth": "", "answer": log.get("answer", "")}
            for log in logs
            if log.get("feedback_type") == "negative" and log.get("question")
        ]

        return negatives[-limit:]

    def collect_rag_outputs(self, test_cases: List[dict]) -> dict:
        """
        テストケースに対してRAGを実行し、RAGAS入力データを生成

        Returns:
            {"question": [...], "answer": [...], "contexts": [...], "ground_truth": [...]}
        """
        rag_service = get_rag_service()

        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for i, case in enumerate(test_cases):
            question = case["question"]
            logger.info(f"[{i+1}/{len(test_cases)}] Processing: {question[:50]}...")

            try:
                result = rag_service.answer_question(question)

                # ソースからコンテキスト抽出
                contexts = []
                for source_type in ["drive", "slack", "other"]:
                    for src in result["sources_by_type"].get(source_type, []):
                        content = src.get("content", "")
                        if content:
                            contexts.append(content)

                # コンテキストが空の場合のフォールバック
                if not contexts:
                    contexts = ["情報なし"]

                data["question"].append(question)
                data["answer"].append(result["answer"])
                data["contexts"].append(contexts)
                data["ground_truth"].append(case.get("ground_truth", ""))

            except Exception as e:
                logger.error(f"Failed to process: {question[:50]}... - {e}")

        logger.info(f"Collected {len(data['question'])} RAG outputs")
        return data

    def save_testcase(self, question: str, ground_truth: str, path: str = None):
        """テストケースを追加"""
        path = path or settings.evaluation_dataset_path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        entry = {"question": question, "ground_truth": ground_truth}
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(f"Added test case: {question[:50]}...")
