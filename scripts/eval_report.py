"""
評価ログ集計レポート
S3またはローカルの採点ログを読み込み、問題点サマリーを出力する
使い方: python scripts/eval_report.py [--days 30]
"""
import sys
import os
import json
import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

LOCAL_EVAL_LOG = "logs/eval_log.jsonl"
S3_EVAL_PREFIX = "eval_logs/"

AXIS_LABELS = {
    "q1_type_understanding": "①質問タイプ理解",
    "q2_direct_answer":      "②質問への直接回答",
    "q3_minimal_info":       "③不要情報の少なさ",
    "q4_internal_context":   "④社内文脈理解",
    "q5_next_action":        "⑤次の行動の妥当性",
}


def load_from_s3(bucket: str, days: int) -> list:
    import boto3
    s3 = boto3.client("s3")
    entries = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=S3_EVAL_PREFIX):
        for obj in page.get("Contents", []):
            try:
                resp = s3.get_object(Bucket=bucket, Key=obj["Key"])
                entry = json.loads(resp["Body"].read().decode("utf-8"))
                ts = datetime.fromisoformat(entry["timestamp"])
                if ts >= cutoff:
                    entries.append(entry)
            except Exception:
                pass
    return entries


def load_from_local(days: int) -> list:
    if not os.path.exists(LOCAL_EVAL_LOG):
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    entries = []
    with open(LOCAL_EVAL_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ts = datetime.fromisoformat(entry["timestamp"])
                if ts >= cutoff:
                    entries.append(entry)
            except Exception:
                pass
    return entries


def load_entries(days: int) -> list:
    bucket = os.environ.get("S3_STATE_BUCKET")
    if bucket:
        return load_from_s3(bucket, days)
    return load_from_local(days)


def build_report(entries: list, days: int) -> str:
    if not entries:
        return f"過去{days}日間の評価ログがありません。"

    total = len(entries)
    scores_sum = defaultdict(int)
    scores_zero = defaultdict(int)  # 0点の件数
    by_type = defaultdict(list)
    low_score_entries = []  # 合計60点未満

    for e in entries:
        scores = e.get("scores", {})
        q_type = e.get("question_type", "unknown")
        total_score = e.get("total_score", 0)

        by_type[q_type].append(total_score)

        for axis in AXIS_LABELS:
            val = scores.get(axis, 0)
            scores_sum[axis] += val
            if val == 0:
                scores_zero[axis] += 1

        if total_score < 60:
            low_score_entries.append(e)

    lines = []
    lines.append(f"=== 評価レポート（過去{days}日間 / {total}件） ===\n")

    # 軸別平均スコア
    lines.append("【軸別 平均スコア / 0点件数】")
    for axis, label in AXIS_LABELS.items():
        avg = scores_sum[axis] / total
        zeros = scores_zero[axis]
        zero_pct = zeros / total * 100
        flag = " ⚠️" if avg < 12 or zero_pct >= 30 else ""
        lines.append(f"  {label}: {avg:.1f}/20  (0点: {zeros}件 / {zero_pct:.0f}%){flag}")

    # 質問タイプ別平均
    lines.append("\n【質問タイプ別 平均スコア】")
    for q_type, type_scores in sorted(by_type.items()):
        avg = sum(type_scores) / len(type_scores)
        flag = " ⚠️" if avg < 60 else ""
        lines.append(f"  {q_type}: {avg:.1f}/100  ({len(type_scores)}件){flag}")

    # 低スコア（60点未満）の具体例
    if low_score_entries:
        lines.append(f"\n【低スコア事例（{len(low_score_entries)}件 / 60点未満）】")
        for e in low_score_entries[:5]:  # 最大5件表示
            q = e.get("question", "")[:60]
            t = e.get("question_type", "?")
            s = e.get("total_score", 0)
            note = e.get("notes", "")
            scores = e.get("scores", {})
            q1 = scores.get("q1_type_understanding", 0)
            lines.append(f"  [{s}点 / {t}] {q}")
            if note:
                lines.append(f"    → {note}")
            if q1 == 0:
                lines.append(f"    → ①質問タイプ誤認（q1=0）")
        if len(low_score_entries) > 5:
            lines.append(f"  ... 他{len(low_score_entries) - 5}件")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30, help="集計対象の日数（デフォルト30日）")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    entries = load_entries(args.days)
    report = build_report(entries, args.days)
    print(report)


if __name__ == "__main__":
    main()
