#!/usr/bin/env python3
"""
lawqa_jp selection 140問を train100 / test40 に分割するスクリプト。
seed固定（デフォルト: 20251128）でシャッフルし、分布を出力する。
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_selection(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "samples" in data:
        samples = data["samples"]
    elif isinstance(data, list):
        samples = data
    else:
        raise ValueError("selection.json は dict(samples) か list を想定しています")

    if len(samples) != 140:
        print(f"[WARN] selection 件数が想定と異なります: len={len(samples)}")
    return list(samples)


def describe_split(name: str, items: List[Dict[str, Any]]) -> None:
    laws = Counter(_get_law_name(x) for x in items)
    labels = Counter(_get_answer(x) for x in items)
    print(f"=== {name} ===")
    print(f"  num_items: {len(items)}")
    print(f"  law_name distribution: {dict(laws)}")
    print(f"  answer distribution: {dict(labels)}")


def _get_law_name(sample: Dict[str, Any]) -> str:
    law = sample.get("law_name")
    if law:
        return str(law)
    fname = sample.get("ファイル名") or sample.get("file_name") or ""
    return fname.split("_")[0] if fname else "UNKNOWN"


def _get_answer(sample: Dict[str, Any]) -> str:
    ans = sample.get("answer") or sample.get("output")
    if isinstance(ans, str):
        return ans.strip().lower()
    return "UNKNOWN"


def split_samples(samples: List[Dict[str, Any]], seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    random.seed(seed)
    shuffled = list(samples)
    random.shuffle(shuffled)
    test_items = shuffled[:40]
    train_items = shuffled[40:]
    if len(test_items) != 40 or len(train_items) != 100:
        raise ValueError(f"split 件数がおかしいです: train={len(train_items)}, test={len(test_items)}")
    return train_items, test_items


def save_split(path: Path, samples: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"samples": samples}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split lawqa_jp selection into train100/test40 with fixed seed.")
    parser.add_argument("--input", type=Path, default=Path("datasets/lawqa_jp/data/selection.json"))
    parser.add_argument("--output-train", type=Path, default=Path("datasets/lawqa_jp/data/selection_train_final100.json"))
    parser.add_argument("--output-test", type=Path, default=Path("datasets/lawqa_jp/data/selection_test_final40.json"))
    parser.add_argument("--seed", type=int, default=20251128)
    args = parser.parse_args()

    samples = load_selection(args.input)
    train_items, test_items = split_samples(samples, seed=args.seed)

    save_split(args.output_train, train_items)
    save_split(args.output_test, test_items)

    print("[INFO] Saved:")
    print(f"  train -> {args.output_train} (len={len(train_items)})")
    print(f"  test  -> {args.output_test} (len={len(test_items)})")
    describe_split("TRAIN_FINAL100", train_items)
    describe_split("TEST_FINAL40", test_items)


if __name__ == "__main__":
    main()
