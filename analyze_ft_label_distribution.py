#!/usr/bin/env python3
"""
ft_direct_full_norag.jsonl のラベル分布を集計するスクリプト。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Count label distribution in ft JSONL.")
    parser.add_argument("-p", "--path", type=Path, default=Path("datasets/ft_direct_full_norag.jsonl"))
    args = parser.parse_args()

    label_all = Counter()
    label_kinsyo = Counter()
    label_yakki = Counter()
    label_shakuchi = Counter()

    for obj in iter_jsonl(args.path):
        file_name = obj.get("file_name", "")
        label = obj.get("correct_answer")
        if label is None:
            continue

        label_all[label] += 1
        if file_name.startswith("金商法"):
            label_kinsyo[label] += 1
        if file_name.startswith("薬機法"):
            label_yakki[label] += 1
        if file_name.startswith("借地借家法"):
            label_shakuchi[label] += 1

    sections = [
        ("全体のラベル分布", label_all),
        ("金商法のラベル分布", label_kinsyo),
        ("薬機法のラベル分布", label_yakki),
        ("借地借家法のラベル分布", label_shakuchi),
    ]

    for title, counter in sections:
        print(f"=== {title} ===")
        for label in ["a", "b", "c", "d"]:
            print(f"{label}: {counter.get(label, 0)}")
        print()


if __name__ == "__main__":
    main()
