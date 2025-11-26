#!/usr/bin/env python3
"""
selection.json を科目ごとに stratified split するユーティリティ。
train/dev/test をおおむね 60/20/20 の比率で出力する（比率は引数で変更可）。
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List


def get_subject(sample: Dict) -> str:
    """ファイル名の先頭トークンを科目ラベルとして取得"""
    fname = sample.get("ファイル名", "")
    if not fname:
        return "unknown"
    return fname.split("_")[0]


def stratified_split(
    samples: List[Dict],
    train_ratio: float,
    dev_ratio: float,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """科目ごとにシャッフルし、train/dev/testに分割"""
    rng = random.Random(seed)
    by_subject: Dict[str, List[Dict]] = {}
    for s in samples:
        by_subject.setdefault(get_subject(s), []).append(s)

    splits = {"train": [], "dev": [], "test": []}
    for subject, items in by_subject.items():
        rng.shuffle(items)
        n = len(items)
        n_train = math.floor(n * train_ratio)
        n_dev = math.floor(n * dev_ratio)
        n_test = n - n_train - n_dev  # 端数はtestに回す

        splits["train"].extend(items[:n_train])
        splits["dev"].extend(items[n_train : n_train + n_dev])
        splits["test"].extend(items[n_train + n_dev :])

    return splits


def save_json(path: Path, samples: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"samples": samples}
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split selection.json into train/dev/test with stratification.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/lawqa_jp/data/selection.json"),
        help="Path to selection.json",
    )
    parser.add_argument(
        "--output-train",
        type=Path,
        default=Path("datasets/lawqa_jp/data/selection_train.json"),
        help="Output path for train split",
    )
    parser.add_argument(
        "--output-dev",
        type=Path,
        default=Path("datasets/lawqa_jp/data/selection_dev.json"),
        help="Output path for dev split",
    )
    parser.add_argument(
        "--output-test",
        type=Path,
        default=Path("datasets/lawqa_jp/data/selection_test.json"),
        help="Output path for test split",
    )
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Train split ratio (default 0.6)")
    parser.add_argument("--dev-ratio", type=float, default=0.2, help="Dev split ratio (default 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    if args.train_ratio + args.dev_ratio > 1.0:
        raise ValueError("train_ratio + dev_ratio must be <= 1.0")

    with args.input.open(encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", [])
    if not samples:
        raise ValueError(f"No samples found in {args.input}")

    splits = stratified_split(samples, args.train_ratio, args.dev_ratio, seed=args.seed)

    save_json(args.output_train, splits["train"])
    save_json(args.output_dev, splits["dev"])
    save_json(args.output_test, splits["test"])

    total = len(samples)
    print(f"Total samples: {total}")
    print(
        f"Train: {len(splits['train'])} ({len(splits['train'])/total:.1%}), "
        f"Dev: {len(splits['dev'])} ({len(splits['dev'])/total:.1%}), "
        f"Test: {len(splits['test'])} ({len(splits['test'])/total:.1%})"
    )


if __name__ == "__main__":
    main()
