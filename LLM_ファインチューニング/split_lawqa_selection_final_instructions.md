# lawqa_jp selection: train100 / test40 split スクリプト作成指示（Codex 向け）

最終評価用に、`lawqa_jp` の selection 140問を **train100 / test40** に分割する Python スクリプトを追加してください。  
このスクリプトは、卒研で使う **最終 split** を決めるためのものです。

---

## 1. 目的・前提

- リポジトリ: **`statutes-rags-FT`**
- 元データ: `datasets/lawqa_jp/data/selection.json`  
  - lawqa_jp の 4択問題が **140問** 入っている想定です。
  - 各要素は、すでに学習・評価で使っている selection 系 JSON と同じ構造（`file_name`, `law_name`, `answer`, `question`, `options`, ...）になっています。

### 1.1 このスクリプトでやりたいこと

1. `selection.json` の 140問を読み込む。
2. **ランダムシード固定（例: `20251128`）** でシャッフルする。
3. シャッフル後の先頭 40問を **test40**、残り 100問を **train100** として分割する。
4. 次の 2ファイルとして保存する：
   - `datasets/lawqa_jp/data/selection_train_final100.json`
   - `datasets/lawqa_jp/data/selection_test_final40.json`
5. 簡単な統計情報（件数、法域・ラベル分布）を標準出力に表示する。

> ※ 重要：**この split は「最終評価用」として固定する前提**です。  
>    一度決めたら、seed やアルゴリズムは変えないでください。

---

## 2. 追加してほしいスクリプトの場所・ファイル名

- 新規ファイル（想定）:  
  **`scripts/split_lawqa_selection_final.py`**

> 既存の構成に合わせて、`scripts/` 配下に置いてください。  
>（別ファイル名でも構いませんが、「何をするスクリプトか」が分かる名前でお願いします）

---

## 3. スクリプトの仕様（詳細）

### 3.1 コマンドライン引数

最低限、以下をサポートしてください（argparse でOK）：

- `--input`
  - デフォルト: `datasets/lawqa_jp/data/selection.json`
- `--output-train`
  - デフォルト: `datasets/lawqa_jp/data/selection_train_final100.json`
- `--output-test`
  - デフォルト: `datasets/lawqa_jp/data/selection_test_final40.json`
- `--seed`
  - デフォルト: `20251128`（整数）

例：

```bash
# ほぼ想定されている実行例
python scripts/split_lawqa_selection_final.py

# seed を変えたい場合（通常は使わない想定）
python scripts/split_lawqa_selection_final.py --seed 12345
```

### 3.2 実装の流れ

擬似コード的には以下のような流れでお願いします。

```python
import argparse
import json
import random
from collections import Counter
from pathlib import Path

def load_selection(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("selection.json は list 形式を想定しています")
    if len(data) != 140:
        # 厳密な値は selection.json の現状に合わせてよいが、
        # 想定と違っていたら warning を出す。
        print(f"[WARN] selection 件数が想定と異なります: len={len(data)}")
    return data

def describe_split(name: str, items):
    # 法域（law_name）と正解ラベル（answer）の分布を表示
    from collections import Counter
    laws = Counter(x.get("law_name", "UNKNOWN") for x in items)
    labels = Counter(x.get("answer", "UNKNOWN") for x in items)
    print(f"=== {name} ===")
    print(f"  num_items: {len(items)}")
    print(f"  law_name distribution: {dict(laws)}")
    print(f"  answer distribution: {dict(labels)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default="datasets/lawqa_jp/data/selection.json")
    parser.add_argument("--output-train", type=str,
                        default="datasets/lawqa_jp/data/selection_train_final100.json")
    parser.add_argument("--output-test", type=str,
                        default="datasets/lawqa_jp/data/selection_test_final40.json")
    parser.add_argument("--seed", type=int, default=20251128)
    args = parser.parse_args()

    input_path = Path(args.input)
    train_path = Path(args.output_train)
    test_path = Path(args.output_test)

    data = load_selection(input_path)

    # 再現性のために seed 固定
    random.seed(args.seed)

    # シャッフル（元データをコピーしてからシャッフルする）
    shuffled = list(data)
    random.shuffle(shuffled)

    # 先頭40問を test、残り100問を train にする
    test_items = shuffled[:40]
    train_items = shuffled[40:]

    if len(test_items) != 40 or len(train_items) != 100:
        raise ValueError(f"split 件数がおかしいです: train={len(train_items)}, test={len(test_items)}")

    # 保存（UTF-8, インデント付き, 日本語可）
    train_path.write_text(json.dumps(train_items, ensure_ascii=False, indent=2), encoding="utf-8")
    test_path.write_text(json.dumps(test_items, ensure_ascii=False, indent=2), encoding="utf-8")

    # 簡単な統計を出力
    print("[INFO] Saved:")
    print(f"  train -> {train_path} (len={len(train_items)})")
    print(f"  test  -> {test_path} (len={len(test_items)})")
    describe_split("TRAIN_FINAL100", train_items)
    describe_split("TEST_FINAL40", test_items)

if __name__ == "__main__":
    main()
```

> 上記はあくまでイメージです。  
> 実際の `selection.json` のキー名（特に `law_name`, `answer`）が違う場合は、  
> 実データに合わせて調整してください。

---

## 4. 実装後に確認してほしいこと

1. スクリプトを実行する：

    ```bash
    python scripts/split_lawqa_selection_final.py
    ```

2. 次のファイルが生成されていること：
    - `datasets/lawqa_jp/data/selection_train_final100.json`
    - `datasets/lawqa_jp/data/selection_test_final40.json`

3. それぞれの長さ：

    ```python
    import json
    from pathlib import Path

    train = json.loads(Path("datasets/lawqa_jp/data/selection_train_final100.json").read_text(encoding="utf-8"))
    test = json.loads(Path("datasets/lawqa_jp/data/selection_test_final40.json").read_text(encoding="utf-8"))
    print(len(train), len(test))  # 100, 40 を想定
    ```

4. コンソール出力の分布情報を確認し、
   - 金商法 / 薬機法 / 借地借家法（law_name）
   - answer（a/b/c/d）
   の分布が「極端に偏っていないか」だけ軽く目視チェックする。

> この split は **最終評価用**として固定するので、  
> 一度問題なさそうなら、以降は seed やアルゴリズムを変えないでください。

---

## 5. この後の流れ（参考）

このスクリプトができたら、次のステップとして：

1. `selection_train_final100.json` を使って、  
   `build_finetune_dataset.py` で **FT v4 用 JSONL** を生成する。
2. heart01(A100) で v4 学習を行う。
3. `selection_test_final40.json` を使って、  
   - ベースライン v0（FTなし）  
   - FT v4  
   を同じ評価スクリプト＋answer parser v3 で評価し、  
   最終レポート用の test40 スコアとして使う。

まずはこの split スクリプトの実装をお願いします。
