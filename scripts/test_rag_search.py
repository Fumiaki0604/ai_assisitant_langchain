"""
RAG検索と回答生成のテスト
"""
import sys
import os

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.rag_service import get_rag_service
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_rag_search():
    """
    RAG検索のテスト
    """
    try:
        # RAGサービスを取得
        service = get_rag_service()

        # テスト質問
        test_questions = [
            "メルカートとは何ですか？",
            "ECサイトの構築について教えてください",
            "提案資料について教えてください",
        ]

        print("\n" + "="*60)
        print("RAG検索テスト開始")
        print("="*60 + "\n")

        for i, question in enumerate(test_questions, 1):
            print(f"\n[質問{i}]: {question}")
            print("-" * 60)

            # RAG検索と回答生成
            result = service.answer_question(question)

            # 回答を表示
            print(f"\n[回答]:\n{result['answer']}")

            # 参考にしたドキュメントを表示
            print(f"\n[参考ドキュメント]: {len(result['sources'])}件")
            for j, src in enumerate(result['sources'], 1):
                print(f"\n  [{j}] {src['title']}")
                print(f"      ソース: {src['source']}")
                print(f"      内容: {src['content'][:80]}...")

            print("\n" + "="*60)

        print("\nRAG検索テスト完了")
        return True

    except Exception as e:
        logger.error(f"エラー: {e}", exc_info=True)
        print(f"\nエラー: {e}\n")
        return False


if __name__ == "__main__":
    test_rag_search()
