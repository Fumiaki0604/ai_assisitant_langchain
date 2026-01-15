"""
フィードバック統計を表示
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feedback.feedback_logger import get_feedback_logger


def main():
    fb_logger = get_feedback_logger()

    # 統計を表示
    stats = fb_logger.get_stats()

    print("\n" + "="*50)
    print("フィードバック統計")
    print("="*50)
    print(f"  合計: {stats['total']} 件")
    print(f"  良い (👍): {stats['positive']} 件")
    print(f"  悪い (👎): {stats['negative']} 件")
    print(f"  満足度: {stats['positive_rate']}%")
    print("="*50)

    # ネガティブフィードバックがあれば表示
    negative_feedbacks = fb_logger.get_negative_feedbacks(limit=5)

    if negative_feedbacks:
        print("\n改善が必要な回答（直近5件）:")
        print("-"*50)

        for i, fb in enumerate(negative_feedbacks, 1):
            print(f"\n[{i}] {fb['timestamp']}")
            print(f"    質問: {fb['question'][:100]}..." if len(fb['question']) > 100 else f"    質問: {fb['question']}")
            print(f"    回答: {fb['answer'][:100]}..." if len(fb['answer']) > 100 else f"    回答: {fb['answer']}")

    print()


if __name__ == "__main__":
    main()
