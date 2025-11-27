# statutes-rags-FT: answer パーサ改善指示（Codex 向け）
_A100 側での no-RAG 直答 v3 / dev 検証を前提とした修正_

このドキュメントは、`scripts/evaluate_multiple_choice.py` で使っている
「LLM 出力から a/b/c/d を取り出す answer パーサ」を改善するための指示です。

目的は、

- `回答（1文字のみ）: a` などをオウム返しするループ出力
- 解説文の中に出てくる a/b/c/d
- `Answer: c` のように行末に正解だけを書いているケース

などを **安定して正しくパースすること** です。

特に dev で確認された以下のような問題を潰すことを狙います：

- モデルが本文で「正解は c です」「回答（1文字のみ）: c」と書いているのに、
  解説の冒頭付近に出てくる `a` を拾ってしまい、`predicted_answer = "a"` になる。
- 複数行にわたって `回答（1文字のみ）: a` を繰り返す中に、
  一度だけ `回答（1文字のみ）: c` が出ているケースに対応できていない。

---

## 1. 修正対象ファイル

- リポジトリ: `statutes-rags-FT`
- ファイル: `scripts/evaluate_multiple_choice.py`

このファイル内にある、

- LLM 出力（`response_text` など）から a/b/c/d を決める処理

を **新しい関数 `normalize_and_parse_answer()` に集約し、
評価ループからはこの関数だけを呼ぶ構造** にしてください。

既に `normalize_and_parse_answer()` が存在する場合は、
ここに書いている仕様に合わせて **中身を置き換える／拡張する** 形で OK です。

---

## 2. 新しいパーサの設計方針（仕様）

### 2-1. 入出力インターフェース

```python
from typing import Optional

def normalize_and_parse_answer(raw_output: str) -> Optional[str]:
    """
    LLM の生出力テキストから、最終的な回答 a/b/c/d を 1 文字で取り出す。
    - 正解候補が見つかれば 'a' / 'b' / 'c' / 'd' を返す（すべて小文字）
    - 見つからなければ None を返す（= unknown 扱い）
    """
    ...
```

### 2-2. 正規化の基本ルール

1. **全角→半角・大文字→小文字** に揃える
   - `unicodedata.normalize("NFKC", raw_output)` を使う
   - その後 `.lower()` する

2. **行単位で処理する**
   - `lines = [ln.strip() for ln in text.splitlines() if ln.strip()]` のように、
     空行を除いた行リストを作る。

3. **「Answer:」「回答:」行を最優先で見る**
   - 最後に出てきた `Answer:` / `回答:` を含む行から、a/b/c/d を抽出する。
   - それでも取れない場合のみ、テキスト全体からのフォールバックに進む。

### 2-3. 文字のマッピング

- 数字 `1`, `2`, `3`, `4` が単独で出てきた場合は、
  - `1 -> 'a'`
  - `2 -> 'b'`
  - `3 -> 'c'`
  - `4 -> 'd'`
  とみなす。
- これらは **正規化後に** 処理する（全角 "１" も "1" になる）。

---

## 3. 具体的なアルゴリズム（擬似コード）

### 3-1. ユーティリティ関数：1文字を a/b/c/d にマップする

```python
import re
import unicodedata
from typing import Optional

CHOICE_MAP = {"1": "a", "2": "b", "3": "c", "4": "d"}
VALID_CHOICES = {"a", "b", "c", "d"}

def _extract_choice_from_text(text: str) -> Optional[str]:
    # 全角・大文字などを正規化
    normalized = unicodedata.normalize("NFKC", text).lower()

    # 1. まず数字 1〜4 を単独で探す
    #    例: "回答: 3" や "3\n回答（1文字のみ）: 3" 等
    m = re.search(r"(?<![0-9])([1-4])(?![0-9])", normalized)
    if m:
        return CHOICE_MAP[m.group(1)]

    # 2. 次に a〜d を単独のトークンとして探す
    #    例: "answer: a", "回答: b" など
    m = re.search(r"(?<![a-z])([abcd])(?![a-z])", normalized)
    if m:
        return m.group(1)

    return None
```

### 3-2. メイン関数：normalize_and_parse_answer

```python
def normalize_and_parse_answer(raw_output: str) -> Optional[str]:
    """
    1. 行単位に分割
    2. 下から上に向かって「Answer: / 回答: を含む行」を探す
    3. 見つかったら、その行から a/b/c/d を取り出す（_extract_choice_from_text を使用）
    4. それでもダメなら、テキスト全体を対象にフォールバック検索
    """
    if not raw_output:
        return None

    # まずはテキスト全体を NFKC + lower で正規化
    text = unicodedata.normalize("NFKC", raw_output).lower()

    # 行分割（空行は除外）
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # ---------- Step 1: "answer" / "回答" を含む行を、末尾から優先的に見る ----------
    for line in reversed(lines):
        if "answer" in line or "回答" in line:
            choice = _extract_choice_from_text(line)
            if choice in VALID_CHOICES:
                return choice

    # ---------- Step 2: 末尾から 2〜3 行分をまとめて見る（オウム返し対策） ----------
    tail = "\n".join(lines[-3:]) if len(lines) >= 3 else "\n".join(lines)
    choice = _extract_choice_from_text(tail)
    if choice in VALID_CHOICES:
        return choice

    # ---------- Step 3: 全体を対象にフォールバック ----------
    choice = _extract_choice_from_text(text)
    if choice in VALID_CHOICES:
        return choice

    # 見つからなければ unknown
    return None
```

ポイント：

- 「最後に出た Answer/回答 行」を最優先するため、`reversed(lines)` を使う。
- オウム返しで `回答: a` `回答: b` `回答: c` … と複数出る場合、
  **下のほう（=より新しい行）を優先**して拾うことになる。
- 解説本文に出てくる `a` や `b` ではなく、`Answer:` や `回答:` をトリガーにすることで、
  誤パースを減らす。

---

## 4. 評価ループ側の修正イメージ

`scripts/evaluate_multiple_choice.py` 内で、
LLM の生出力（例：`raw_answer_text`）から `predicted_answer` を決めている箇所を、
すべて `normalize_and_parse_answer()` を通すようにしてください。

### 4-1. 変更前（イメージ）

```python
raw_output = result["raw_output"]  # 例
# どこかで単純に 'a'/'b'/'c'/'d' を探している処理
predicted_answer = simple_parse_answer(raw_output)
if predicted_answer not in ["a", "b", "c", "d"]:
    predicted_answer = "unknown"
```

### 4-2. 変更後（イメージ）

```python
raw_output = result["raw_output"]  # LLM の完全な出力テキスト

choice = normalize_and_parse_answer(raw_output)
if choice is None:
    predicted_answer = "unknown"
else:
    predicted_answer = choice
```

- `unknown` のカウントロジック等は従来どおりで OK です。
- すでに `normalize_and_parse_answer` を使っている場合は、上記仕様にあわせて
  フォールバックや正規化の挙動をアップデートしてください。

---

## 5. 動作確認用のテストケース（簡易）

実装後、Python REPL か簡単なテストスクリプトで、
以下のような文字列を `normalize_and_parse_answer` に渡して、
期待どおりにパースできるか確認してください。

### ケース1: シンプルな Answer 行

```python
s = "問題文....\n\nAnswer: c"
assert normalize_and_parse_answer(s) == "c"
```

### ケース2: 「回答（1文字のみ）」と数字の組み合わせ

```python
s = "回答（1文字のみ）: ３"
assert normalize_and_parse_answer(s) == "c"
```

### ケース3: 解説の中に a/b/c/d が出てくるが、最後の Answer 行が正しい

```python
s = "(1) 選択肢aは条文の要件を満たさない。\n(2) 選択肢bも不適切である。\n以上より、正解はcである。\n\nAnswer: c"
assert normalize_and_parse_answer(s) == "c"
```

### ケース4: オウム返しで複数行に渡る出力

```python
s = "回答（1文字のみ）: a\n回答（1文字のみ）: b\n回答（1文字のみ）: c\n回答（1文字のみ）: c"
assert normalize_and_parse_answer(s) == "c"
```

### ケース5: 何も a/b/c/d/1/2/3/4 が見つからない場合

```python
s = "よく分かりません。"
assert normalize_and_parse_answer(s) is None
```

---

## 6. この修正によって期待されること

- dev で観測されたような、
  - 「モデルは c と書いているのに、評価が a になってしまう」
  - 「Answer 行を無視して、本文中の a を拾ってしまう」
- といった **パーサ由来の誤判定が減り、
  v3 の真の精度（特に dev）が少し上振れする** ことが期待されます。

実装後は、A100 側で

- v0（FTなし）
- v2（旧FT）
- v3（新FT）

の dev 評価をこの新パーサで再実行し、
精度の変化を比較できるようにしてもらえると助かります。
