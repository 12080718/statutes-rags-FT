# statutes-rags-FT: answer パーサ v3 改善指示（Codex 向け）
_A100 側での no-RAG 直答 v3 / dev/test 検証を前提とした修正_

このドキュメントは、`scripts/evaluate_multiple_choice.py` で使っている
「LLM 出力から a/b/c/d を取り出す answer パーサ」を **v3 仕様** に更新するための指示です。

目的は、

- `回答（1文字のみ）: a` などの回答行
- `Answer: c` のような英語の回答行
- `正解は a です。` のような日本語の解説
- 末尾に 1 文字だけ `a` や `3` と書いてある行

といったパターンを **優先順位つきで安定してパース** することです。

特に、dev ログで確認された以下の問題を解消することを狙います：

- モデルが本文で「正解は a です」「回答（1文字のみ）: b」と書いているのに、
  解説の冒頭付近に出てくる `a` を拾ってしまい、`predicted_answer` が誤る。
- 同じ `response` に対して、パーサの変更だけで精度が 19/27 → 15/27 に悪化してしまう。

---

## 1. 修正対象ファイル

- リポジトリ: `statutes-rags-FT`
- ファイル: `scripts/evaluate_multiple_choice.py`

このファイル内にある、

- LLM 出力（`response_text` など）から a/b/c/d を決める処理

を **`normalize_and_parse_answer()` 関数（v3仕様）に集約**し、
評価ループからはこの関数だけを呼ぶ構造にしてください。

すでに `normalize_and_parse_answer()` が存在する場合は、
ここに書いている v3 仕様に合わせて **中身を置き換える／拡張する** 形で OK です。

---

## 2. v3 パーサの設計方針（概要）

### 2-1. 目標

1. **LLM が明示的に書いた「回答行」を最優先で読む**
   - 例: `回答（1文字のみ）: b`
   - 例: `Answer: c`
2. 回答行が無い場合だけ、
   - `正解は a です。` のような行
   - 末尾に 1 文字だけ `a` や `3` と書かれた行
   にフォールバックする。
3. それでも決まらない場合は **None（unknown）** を返す。

### 2-2. やらないこと

- テキスト全体から無差別に `a` / `b` / `1` / `2` を拾わない。
  - 解説中の `選択肢a` や `第2条第1項第15号` などを誤って拾わないようにする。
- `answer` / `回答` / `正解は` などのトリガーなしに、
  単に「最初に見つかった `a`」を採用しない。

---

## 3. インターフェースと前処理

### 3-1. 関数インターフェース

```python
from typing import Optional

def normalize_and_parse_answer(raw_output: str) -> Optional[str]:
    """
    LLM の生出力テキストから、最終的な回答 a/b/c/d を 1 文字で取り出す（v3仕様）。
    - 正解候補が見つかれば 'a' / 'b' / 'c' / 'd' を返す（小文字）
    - 見つからなければ None を返す（= unknown 扱い）
    """
    ...
```

### 3-2. 前処理（正規化と行分割）

```python
import unicodedata

def normalize_and_parse_answer(raw_output: str) -> Optional[str]:
    if not raw_output:
        return None

    # 全角→半角、A→a 等を揃える
    text = unicodedata.normalize("NFKC", raw_output).lower()

    # 行ごとに分割し、空行は除外
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # 以下、lines を使って各ステップを実行する
    ...
```

---

## 4. トークン抽出ユーティリティ

### 4-1. a/b/c/d と 1/2/3/4 のマッピング

```python
import re
from typing import Optional, List

CHOICE_MAP = {"1": "a", "2": "b", "3": "c", "4": "d"}
VALID_CHOICES = {"a", "b", "c", "d"}

def _extract_tokens(seg: str) -> List[str]:
    """
    正規化済みテキスト seg から、a/b/c/d または 1/2/3/4 を
    「単独トークン」としてすべて抜き出す。
    """
    # (?<![0-9a-z]) と (?![0-9a-z]) で、前後が英数字でない「単独文字」に限定
    return re.findall(r"(?<![0-9a-z])([1-4abcd])(?![0-9a-z])", seg)

def _choice_from_tokens(tokens: List[str]) -> Optional[str]:
    """
    トークン列から、最後に出てきた 1/2/3/4 or a/b/c/d を a/b/c/d に変換して返す。
    """
    if not tokens:
        return None

    # 一番「後ろ」に出てきたトークンを採用
    t = tokens[-1]

    if t in CHOICE_MAP:
        return CHOICE_MAP[t]
    if t in VALID_CHOICES:
        return t
    return None
```

---

## 5. v3 の探索手順（優先度付き）

### Step 1: 「回答」or「answer」を含む行（最優先）

- 下から上に向かって（`reversed(lines)`）、
  - `if "answer" in ln or "回答" in ln:` を満たす行を探す。
- 見つかった行に `_extract_tokens` をかけて、
  `_choice_from_tokens` で a/b/c/d を決める。
- 最初に決まったものを即 `return` する。

```python
    # Step 1: 回答/answer を含む行（末尾から）
    for ln in reversed(lines):
        if "answer" in ln or "回答" in ln:
            tokens = _extract_tokens(ln)
            choice = _choice_from_tokens(tokens)
            if choice in VALID_CHOICES:
                return choice
```

> これで  
> `回答（1文字のみ）: b` や `answer: c` の **一番下の行** が優先的に拾われます。

---

### Step 2: 「正解は〜」系の行

- まだ決まっていない場合、
  - 同様に下から上に走査し、
  - `if "正解は" in ln or "正解が" in ln:` を満たす行を探す。
- 見つかった行から `_extract_tokens` で a/b/c/d or 1〜4 を取り出し、
  `_choice_from_tokens` で決める。

```python
    # Step 2: 「正解は〜」系の行（末尾から）
    for ln in reversed(lines):
        if "正解は" in ln or "正解が" in ln:
            tokens = _extract_tokens(ln)
            choice = _choice_from_tokens(tokens)
            if choice in VALID_CHOICES:
                return choice
```

> これで  
> `以上より、正解は a である。`  
> といった行から a が拾えます。

---

### Step 3: 末尾数行の「1文字だけの行」にフォールバック

- それでも決まらない場合、
  - 末尾 5 行程度を対象に、
  - 1 行の中に `a/b/c/d` または `1/2/3/4` しか含まれていない行がないかを見る。
- 同じ文字が連続しているだけなら、その文字を採用する。

```python
    # Step 3: 末尾5行の「1文字だけの行」
    import re as _re  # ファイル先頭ですでに import しているなら不要

    tail_lines = lines[-5:] if len(lines) > 5 else lines
    for ln in reversed(tail_lines):
        # a〜d と 1〜4 以外を削る
        stripped = _re.sub(r"[^abcd1-4]", "", ln)
        if not stripped:
            continue
        uq = set(stripped)
        if len(uq) != 1:
            continue  # 混在している場合は無視
        ch = next(iter(uq))
        if ch in CHOICE_MAP:
            return CHOICE_MAP[ch]
        if ch in VALID_CHOICES:
            return ch
```

> これで、解説のあとに素で `a` / `3` とだけ書くような出力も拾えます。

---

### Step 4: 何も取れない場合

```python
    # どのパターンでも取れなかった場合は unknown 扱い
    return None
```

評価側では、`None` を受け取ったら `predicted_answer = "unknown"` のように扱ってください。

---

## 6. 評価ループ側の修正イメージ

`scripts/evaluate_multiple_choice.py` 内で、
LLM の生出力（例：`raw_answer_text`）から `predicted_answer` を決めている箇所を、
すべて `normalize_and_parse_answer()`（v3仕様）を通すようにしてください。

### 変更例

```python
raw_output = result["raw_output"]  # LLM の完全な出力テキスト

choice = normalize_and_parse_answer(raw_output)
if choice is None:
    predicted_answer = "unknown"
else:
    predicted_answer = choice
```

- `unknown` のカウントロジック等は従来どおりで構いません。
- 旧 `normalize_and_parse_answer` を使っている場合は、上記 v3 仕様に置き換えてください。

---

## 7. 動作確認用のテストケース（簡易）

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

### ケース4: オウム返しで複数行に渡る出力（最後が正とみなす）

```python
s = "回答（1文字のみ）: a\n回答（1文字のみ）: b\n回答（1文字のみ）: c\n回答（1文字のみ）: c"
assert normalize_and_parse_answer(s) == "c"
```

### ケース5: 「正解は〜」しかない場合

```python
s = "選択肢dは不適切である。\nしたがって、正解は b である。"
assert normalize_and_parse_answer(s) == "b"
```

### ケース6: 何も a/b/c/d/1/2/3/4 が見つからない場合

```python
s = "よく分かりません。"
assert normalize_and_parse_answer(s) is None
```

---

## 8. 実装後にやってほしいこと

1. この v3 パーサを組み込んだ状態で、A100 側で
   - v0（FTなし）
   - v2（旧FT）
   - v3（新FT）
   の dev/test 評価を再実行し、精度を比較してください。

2. 特に dev30 については、
   - 旧ログ（parser変更前）と新ログ（parser v3）の `predicted_answer` を突き合わせて、
   - 「モデルが本当に変わっていないこと」
   - 「パーサだけを変えたときの精度の変化」
   を確認できると、卒研の分析・考察にも使いやすくなります。
