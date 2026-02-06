"""
テストセット作成ヘルパー
対話式でテストケースをtestset.jsonlに追加
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.dataset_manager import DatasetManager
from config.settings import settings


def interactive_mode(path: str):
    """対話式でテストケースを追加"""
    dm = DatasetManager()

    print("テストケース作成（Ctrl+Cで終了）")
    print("=" * 50)

    count = 0
    try:
        while True:
            print(f"\n--- ケース {count + 1} ---")
            question = input("質問: ").strip()
            if not question:
                continue

            ground_truth = input("期待回答: ").strip()
            if not ground_truth:
                print("期待回答は必須です。スキップします。")
                continue

            dm.save_testcase(question, ground_truth, path)
            count += 1
            print(f"追加しました（累計: {count} 件）")

    except (KeyboardInterrupt, EOFError):
        print(f"\n\n完了: {count} 件のテストケースを追加しました → {path}")


def import_csv(csv_path: str, output_path: str):
    """CSVからインポート（ヘッダー: question,ground_truth）"""
    import csv

    dm = DatasetManager()
    count = 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row.get("question", "").strip()
            ground_truth = row.get("ground_truth", "").strip()
            if question and ground_truth:
                dm.save_testcase(question, ground_truth, output_path)
                count += 1

    print(f"インポート完了: {count} 件 → {output_path}")


def show_testset(path: str):
    """既存テストセットを表示"""
    dm = DatasetManager()
    cases = dm.load_testset(path)

    if not cases:
        print(f"テストセットが空です: {path}")
        return

    print(f"テストセット: {len(cases)} 件 ({path})")
    print("=" * 50)
    for i, case in enumerate(cases, 1):
        print(f"\n{i}. Q: {case['question']}")
        print(f"   A: {case['ground_truth']}")


def main():
    parser = argparse.ArgumentParser(description='テストセット作成')
    parser.add_argument('--path', default=None, help='テストセットのパス')
    parser.add_argument('--import-csv', default=None, help='CSVからインポート')
    parser.add_argument('--show', action='store_true', help='既存テストセットを表示')
    args = parser.parse_args()

    path = args.path or settings.evaluation_dataset_path

    if args.show:
        show_testset(path)
    elif args.import_csv:
        import_csv(args.import_csv, path)
    else:
        interactive_mode(path)


if __name__ == "__main__":
    main()
