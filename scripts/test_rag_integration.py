"""
RAG統合のテスト（Slackボットを起動せずにRAGサービスのみテスト）
"""
import sys
import os

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.rag_service import get_rag_service
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_rag_integration():
    """
    RAG統合テスト
    """
    try:
        print("\n" + "="*60)
        print("RAG統合テスト開始")
        print("="*60 + "\n")

        # RAGサービスを取得
        rag_service = get_rag_service()

        # テスト質問
        test_questions = [
            "リモートワークは何曜日でもできますか？",
            "タクシー使いたいんですが",
        ]

        for i, question in enumerate(test_questions, 1):
            print(f"\n【質問{i}】: {question}")
            print("-" * 60)

            # RAGで回答生成
            result = rag_service.answer_question(question)

            # 回答を表示
            print(f"\n【回答】:\n{result['answer']}")

            # 参考ドキュメント表示
            if result['sources']:
                print(f"\n【参考ドキュメント】:")
                for j, source in enumerate(result['sources'][:2], 1):
                    print(f"  • {source['title']}")

            print("\n" + "="*60)

        print("\nRAG統合テスト完了")
        return True

    except Exception as e:
        logger.error(f"エラー: {e}", exc_info=True)
        print(f"\nエラー: {e}\n")
        return False


if __name__ == "__main__":
    test_rag_integration()
