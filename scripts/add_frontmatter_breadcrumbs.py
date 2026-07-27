#!/usr/bin/env python3
"""
mercartドキュメントにYAMLフロントマターとパンくずを一括追加するスクリプト
"""
import re
from pathlib import Path

MERCART_DIR = Path(__file__).parent.parent / "documents" / "mercart"


def parse_filename(filename: str) -> tuple:
    """ファイル名からカテゴリとサブカテゴリを抽出"""
    stem = Path(filename).stem
    parts = stem.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return stem, ""


def process_file(file_path: Path):
    content = file_path.read_text(encoding="utf-8")

    if content.startswith("---"):
        print(f"SKIP (already has frontmatter): {file_path.name}")
        return

    category, subcategory = parse_filename(file_path.name)

    frontmatter = f"---\nsource: mercart\ncategory: {category}\nsubcategory: {subcategory}\n---\n\n"

    lines = content.splitlines(keepends=True)
    new_lines = []
    current_h3 = None

    for line in lines:
        new_lines.append(line)

        h3_match = re.match(r'^### (.+)', line.rstrip())
        if h3_match:
            current_h3 = h3_match.group(1)
            new_lines.append(f"> 分類: メルカート › {category} › {subcategory} › {current_h3}\n")
            continue

        h4_match = re.match(r'^#### (.+)', line.rstrip())
        if h4_match:
            h4_name = h4_match.group(1)
            parent = f" › {current_h3}" if current_h3 else ""
            new_lines.append(f"> 分類: メルカート › {category} › {subcategory}{parent} › {h4_name}\n")

    file_path.write_text(frontmatter + "".join(new_lines), encoding="utf-8")
    print(f"OK: {file_path.name}")


def main():
    files = sorted(MERCART_DIR.glob("*.md"))
    print(f"処理対象: {len(files)} ファイル\n")
    for f in files:
        process_file(f)
    print(f"\n完了: {len(files)} ファイル処理")


if __name__ == "__main__":
    main()
