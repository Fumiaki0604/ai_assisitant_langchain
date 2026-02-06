"""
RAG評価スクリプト
RAGASを使用してRAGパイプラインの精度を評価
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.dataset_manager import DatasetManager
from src.evaluation.ragas_evaluator import RAGASEvaluator
from config.settings import settings
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='RAG評価')
    parser.add_argument('--dataset', default=None, help='テストセットのパス（JSONL）。"feedback"でネガティブフィードバックを使用')
    parser.add_argument('--limit', type=int, default=None, help='評価するケース数の上限')
    parser.add_argument('--output', default=None, help='結果出力ディレクトリ')
    parser.add_argument('--batch-size', type=int, default=10, help='バッチサイズ')
    args = parser.parse_args()

    dm = DatasetManager()

    # テストケースの読み込み
    if args.dataset == "feedback":
        logger.info("Loading negative feedback cases...")
        test_cases = dm.load_negative_feedbacks(limit=args.limit or 20)
        if not test_cases:
            print("ネガティブフィードバックがありません")
            return
    else:
        dataset_path = args.dataset or settings.evaluation_dataset_path
        test_cases = dm.load_testset(dataset_path)
        if not test_cases:
            print(f"テストセットが見つかりません: {dataset_path}")
            print(f"  scripts/create_testset.py でテストケースを作成してください")
            return

    if args.limit:
        test_cases = test_cases[:args.limit]

    print(f"\n評価開始: {len(test_cases)} ケース")
    print("=" * 50)

    # RAG出力を収集
    logger.info("Collecting RAG outputs...")
    evaluation_data = dm.collect_rag_outputs(test_cases)

    if not evaluation_data["question"]:
        print("評価データの収集に失敗しました")
        return

    # RAGAS評価を実行
    logger.info("Running RAGAS evaluation...")
    evaluator = RAGASEvaluator()
    results = evaluator.evaluate(evaluation_data, batch_size=args.batch_size)

    # 結果を保存
    output_path = evaluator.save_results(results, args.output)

    # サマリ表示
    print("\n" + "=" * 50)
    print("評価完了")
    print("=" * 50)

    summary = results.get("summary", {})
    metric_labels = {
        "faithfulness": "Faithfulness（忠実性）",
        "answer_relevancy": "Answer Relevancy（回答関連性）",
        "context_precision": "Context Precision（検索精度）",
        "context_recall": "Context Recall（検索網羅性）",
    }

    for key, label in metric_labels.items():
        s = summary.get(key, {})
        mean = s.get("mean", 0)
        print(f"  {label}: {mean:.2%}")

    # 低スコアケース
    low_cases = [
        c for c in results.get("per_case", [])
        if any(c.get(m, 1.0) < 0.5 for m in metric_labels)
    ]
    if low_cases:
        print(f"\n要改善ケース: {len(low_cases)} 件")
        for case in low_cases[:5]:
            print(f"  - {case['question'][:60]}")

    print(f"\n結果ファイル: {output_path}")


if __name__ == "__main__":
    main()
