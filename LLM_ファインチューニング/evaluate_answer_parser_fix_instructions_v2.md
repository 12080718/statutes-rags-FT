# statutes-rags-FT: no-RAG直答 FT v2 用のコード修正指示（answer パーサ改善）

## 目的

- `scripts/evaluate_multiple_choice.py` 内で、Qwen の出力から a/b/c/d を抽出するロジックが厳しすぎて、
  - 全角文字（例: `ｂ`）や数字（例: `4`）を **すべて `unknown` 扱い** している。
- その結果、本来は正解しているサンプルが `unknown` に落ちてスコアが下がっている。
- これを修正して、
  - **全角→半角の正規化**
  - **1〜4 → a〜d の変換**
  - **「回答: X」「Answer: X」行の優先パース**
  を行う関数を追加し、全体で使う。

---

## 1. 対象ファイル

- `scripts/evaluate_multiple_choice.py`

このファイル内で「モデル出力テキストから a/b/c/d を取り出している箇所」を修正する。

---

## 2. 追加する import

ファイル先頭付近の import 群に、次の2行がなければ追加する。

```python
import unicodedata
import re
```

---

## 3. 新しいパース関数の追加

`evaluate_multiple_choice.py` 内に、次の関数を追加する。

- 追加場所の例:
  - 既存の `parse_answer` や、それに相当する関数定義があればその直前 or 直後。
  - もし `parse_answer` がすでにある場合は **置き換え** でもよい。

```python
def normalize_and_parse_answer(raw_text: str) -> str:
    """
    モデル出力から a/b/c/d を 1 文字で抽出するヘルパー。

    - 全角→半角 (NFKC) で正規化する
    - アルファベットは小文字にそろえる
    - 数字 1〜4 を a〜d にマッピングする
    - 「回答: X」「Answer: X」形式があればそこを優先する
    - それ以外の場合は、末尾側から最初に現れた a/b/c/d/1/2/3/4 を採用する
    - 何も見つからなければ "unknown" を返す
    """
    if not raw_text:
        return "unknown"

    # 全角→半角などの正規化
    text = unicodedata.normalize("NFKC", str(raw_text))
    t = text.lower()

    # 数字→選択肢のマッピング
    digit_to_option = {"1": "a", "2": "b", "3": "c", "4": "d"}

    # ① 「回答: X」「Answer: X」形式を優先
    m = re.search(r"(?:回答|answer)\s*[:：]\s*([abcd1234])", t)
    if m:
        ch = m.group(1)
        if ch in digit_to_option:
            return digit_to_option[ch]
        return ch  # a〜d のいずれか

    # ② テキスト末尾側から走査して、最初に見つかった a/b/c/d/1/2/3/4 を採用
    for ch in reversed(t):
        if ch in "abcd":
            return ch
        if ch in digit_to_option:
            return digit_to_option[ch]

    # ③ それでも見つからなければ unknown
    return "unknown"
```

---

## 4. 既存のパース処理をこの関数に差し替え

`evaluate_multiple_choice.py` 内で、モデルの出力テキストから答えを取り出している箇所を探す。

例（実際のコードは類似するイメージ）：

```python
# 例: 旧実装のイメージ
predicted_answer = parse_answer(model_response_text)
```

または

```python
predicted_answer = extract_answer_from_text(model_response_text)
```

のような行があるはずなので、それらの呼び出しをすべて次のように置き換える。

```python
predicted_answer = normalize_and_parse_answer(model_response_text)
```

- `model_response_text` に相当する変数は、実際の変数名（例: `response_text`, `raw_output` など）に合わせて使う。
- もし `parse_answer` 関数があり、その中に直接正規表現が書かれている場合は、
  - その中身を丸ごと `normalize_and_parse_answer` の実装に差し替えてもよい。

---

## 5. 動作確認の目安

修正後、以下を確認する。

1. 以前 `predicted_answer = "unknown"` になっていたサンプルのうち、
   - 出力中に `ｂ` や `4` のような文字が含まれていたものが、
   - `b` / `d` として正しくカウントされるようになっていること。
2. `scripts/evaluate_multiple_choice.py` を使った既存の評価コマンドがエラーなく動作すること。
   - 例: `python scripts/evaluate_multiple_choice.py --data datasets/lawqa_jp/data/selection.json ...`

---

## 6. メモ

- この修正により、FT の学習自体は変えずに、評価時の `unknown` を減らし、
  本来の性能差をより正確に測れるようにする。
- 卒研のゴール「no-RAG直答ベースライン 42.9% を確実に超えるか？」を判断するうえで重要な修正。
