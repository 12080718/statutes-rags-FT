#!/usr/bin/env python3
"""
薬機法かつ正解ラベルがb/dのサンプルを抽出して表示するスポットチェック用スクリプト。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def truncate(text: str, length: int) -> str:
    if len(text) <= length:
        return text
    return text[:length] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Spot-check yakki b/d samples from ft JSONL.")
    parser.add_argument("-p", "--path", type=Path, default=Path("datasets/ft_direct_full_norag.jsonl"))
    parser.add_argument("-n", "--max-samples", type=int, default=10)
    args = parser.parse_args()

    samples = []
    for obj in iter_jsonl(args.path):
        file_name = obj.get("file_name", "")
        label = obj.get("correct_answer")
        if not file_name.startswith("薬機法"):
            continue
        if label not in {"b", "d"}:
            continue

        question_text = obj.get("question") or obj.get("input", "")
        output_text = obj.get("output", "")
        samples.append(
            {
                "file_name": file_name,
                "label": label,
                "question": truncate(str(question_text), 120),
                "output": truncate(str(output_text), 200),
            }
        )
        if len(samples) >= args.max_samples:
            break

    print("=== 薬機法 b/d 問題のサンプル ===")
    for s in samples:
        print("-" * 40)
        print(f"file_name : {s['file_name']}")
        print(f"label    : {s['label']}")
        print(f"question : {s['question']}")
        print(f"output   : {s['output']}")
    if samples:
        print("-" * 40)


if __name__ == "__main__":
    main()
