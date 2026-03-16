"""
回答品質の自動評価・ログ記録
採点フレーム（100点 / 5軸 × 20点）を使ったLLM評価
"""
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

LOCAL_EVAL_LOG = "logs/eval_log.jsonl"
S3_EVAL_PREFIX = "eval_logs/"

EVAL_PROMPT = """以下のAI回答を採点してください。

## 質問
{question}

## AIの回答
{answer}

## 採点手順

【STEP 1】まず、この質問の「正しいタイプ」を以下の定義で独自に判定してください。
システムが分類した結果は参考にせず、質問文のみから判断すること。

- knowledge  : 知識・仕様・方法を問う質問。例「GA4のイベント設定は？」
- experience : 社内実績・経験者・実装知見を探す質問。
  シグナル：「どなたかご存知ですか」「実装したことある方」「事例ある？」
  「〜わかりません、誰か教えて」「〜を集めてます」「〜担当したことある方」
- document   : 社内資料・テンプレートを持つ人を探す質問。例「〜資料お持ちの方」
- owner      : 特定案件・顧客・業務の担当者を探す質問。例「PDC担当の方いますか？」
- opinion    : 意見・感想を求める質問。例「これどう思う？」

【STEP 2】判定したタイプをもとに、以下の基準で採点してください（各軸 0/5/10/15/20 点）

①質問タイプ理解（20点）
- AIの回答スタイルがSTEP 1で判定した正しいタイプに合っているか
- experience/owner/document なのに一般的な知識説明をしていたら 0点
- 正しいタイプに沿った応答なら 20点

②質問への直接回答（20点）
- 質問の核心に直接答えているか
- 「ある/ない/〜です」という直接回答が冒頭にあるか

③不要情報の少なさ（20点）
- 余分な説明・トレンド解説・外部URL・一般論がないか
- 必要最低限の回答: 20点、大量の不要情報: 0点

④社内文脈理解（20点）
- Slack文化・社内会話トーンに合っているか
- 社内向けに「外部の開発元サポートに問い合わせる」を勧めるのは大幅減点

⑤次の行動の妥当性（20点）
- 質問者が次に取るべき行動を適切に示しているか
- experience/owner/document では社内での次の一手を示す or 提案なしで満点

以下のJSON形式のみで回答（他の文字列は不要）:
{{"correct_type": "判定したタイプ", "q1": 点数, "q2": 点数, "q3": 点数, "q4": 点数, "q5": 点数, "notes": "一言コメント"}}"""


def _get_s3_bucket() -> str | None:
    return os.environ.get("S3_STATE_BUCKET")


def _save_to_s3(bucket: str, entry: dict):
    try:
        import boto3
        s3 = boto3.client("s3")
        ts = entry["timestamp"].replace(":", "-").replace(".", "-")
        channel = entry.get("channel", "unknown")
        key = f"{S3_EVAL_PREFIX}{entry['date']}/{ts}_{channel}.json"
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(entry, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json"
        )
        logger.info(f"Eval log saved to s3://{bucket}/{key}")
    except Exception as e:
        logger.error(f"Failed to save eval log to S3: {e}")


def _save_to_local(entry: dict):
    Path(LOCAL_EVAL_LOG).parent.mkdir(parents=True, exist_ok=True)
    with open(LOCAL_EVAL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _parse_scores(raw: str) -> dict | None:
    try:
        match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group())
        scores = {}
        for k in ("q1", "q2", "q3", "q4", "q5"):
            scores[k] = max(0, min(20, int(data.get(k, 0))))
        scores["notes"] = str(data.get("notes", ""))
        scores["correct_type"] = str(data.get("correct_type", "unknown"))
        scores["total"] = sum(scores[k] for k in ("q1", "q2", "q3", "q4", "q5"))
        return scores
    except Exception as e:
        logger.warning(f"Failed to parse eval scores: {e}")
        return None


def evaluate_and_log(
    question: str,
    answer: str,
    question_type: str,
    channel: str = "",
    user: str = "",
    thread_ts: str = "",
):
    """回答を評価しログを保存。エラーは握りつぶして本体処理を妨げない。"""
    try:
        from src.llm.bedrock import get_bedrock_llm
        llm = get_bedrock_llm()

        prompt = EVAL_PROMPT.format(
            question=question[:500],
            answer=answer[:1000],
        )
        raw = llm.invoke(prompt)
        if hasattr(raw, "content"):
            raw = raw.content

        scores = _parse_scores(raw)
        if not scores:
            logger.warning("Eval score parsing failed, skipping log")
            return

        now = datetime.now(timezone.utc)
        entry = {
            "timestamp": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "channel": channel,
            "user": user,
            "thread_ts": thread_ts,
            "question": question[:500],
            "answer": answer[:1000],
            "classified_type": question_type,
            "correct_type": scores["correct_type"],
            "type_mismatch": question_type != scores["correct_type"],
            "scores": {
                "q1_type_understanding": scores["q1"],
                "q2_direct_answer": scores["q2"],
                "q3_minimal_info": scores["q3"],
                "q4_internal_context": scores["q4"],
                "q5_next_action": scores["q5"],
            },
            "total_score": scores["total"],
            "notes": scores["notes"],
        }

        bucket = _get_s3_bucket()
        if bucket:
            _save_to_s3(bucket, entry)
        else:
            _save_to_local(entry)

        mismatch = " [TYPE MISMATCH]" if question_type != scores["correct_type"] else ""
        logger.info(f"Eval logged: total={scores['total']}/100, classified={question_type}, correct={scores['correct_type']}{mismatch}, q1={scores['q1']}")

    except Exception as e:
        logger.error(f"evaluate_and_log failed: {e}")
