"""
フィードバックログ管理
ボットの回答に対する👍/👎リアクションを記録
"""
import sys
import os
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class FeedbackLogger:
    """
    フィードバックをJSONファイルに記録するクラス
    """

    def __init__(self, log_file: str = None):
        self.log_file = log_file or settings.feedback_log_file

        # ログディレクトリを作成
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # ファイルが存在しない場合は空のリストで初期化
        if not os.path.exists(self.log_file):
            self._save_logs([])

    def _load_logs(self) -> list:
        """ログファイルを読み込み"""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_logs(self, logs: list):
        """ログファイルに保存"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

    def log_feedback(self, feedback_type: str, question: str, answer: str,
                     channel: str, user: str, message_ts: str,
                     sources: list = None):
        """
        フィードバックを記録

        Args:
            feedback_type: "positive" (👍) or "negative" (👎)
            question: 元の質問
            answer: ボットの回答
            channel: チャンネルID
            user: フィードバックしたユーザーID
            message_ts: メッセージのタイムスタンプ
            sources: 参考にしたドキュメント
        """
        logs = self._load_logs()

        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "feedback_type": feedback_type,
            "question": question,
            "answer": answer,
            "channel": channel,
            "user": user,
            "message_ts": message_ts,
            "sources": sources or []
        }

        logs.append(feedback_entry)
        self._save_logs(logs)

        logger.info(f"Feedback logged: {feedback_type} for message {message_ts}")

    def get_stats(self) -> dict:
        """
        フィードバック統計を取得
        """
        logs = self._load_logs()

        positive = sum(1 for log in logs if log["feedback_type"] == "positive")
        negative = sum(1 for log in logs if log["feedback_type"] == "negative")
        total = len(logs)

        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "positive_rate": round(positive / total * 100, 1) if total > 0 else 0
        }

    def get_negative_feedbacks(self, limit: int = 10) -> list:
        """
        ネガティブフィードバックを取得（改善対象）
        """
        logs = self._load_logs()
        negative_logs = [log for log in logs if log["feedback_type"] == "negative"]
        return negative_logs[-limit:]


# シングルトンインスタンス
_feedback_logger_instance = None


def get_feedback_logger() -> FeedbackLogger:
    """
    FeedbackLoggerのシングルトンインスタンスを取得
    """
    global _feedback_logger_instance

    if _feedback_logger_instance is None:
        _feedback_logger_instance = FeedbackLogger()

    return _feedback_logger_instance


if __name__ == "__main__":
    # テスト
    logging.basicConfig(level=logging.INFO)

    fb_logger = get_feedback_logger()

    # テストデータを追加
    fb_logger.log_feedback(
        feedback_type="positive",
        question="リモートワークは週に何日まで可能ですか？",
        answer="週3日までリモートワーク可能です。",
        channel="CME3BV4PN",
        user="U12345678",
        message_ts="1234567890.123456",
        sources=[{"title": "勤務時間ポリシー"}]
    )

    # 統計を表示
    stats = fb_logger.get_stats()
    print(f"\nフィードバック統計:")
    print(f"  合計: {stats['total']} 件")
    print(f"  良い: {stats['positive']} 件")
    print(f"  悪い: {stats['negative']} 件")
    print(f"  良い率: {stats['positive_rate']}%")
