# Codex 用指示書: `ft_direct_full_norag.jsonl` ラベル調査スクリプト

## 目的

法律 QA 用のファインチューニングデータ  
`ft_direct_full_norag.jsonl` に対して、以下 2 つの調査用スクリプトを Python で作成してください。

1. **ラベル分布の集計スクリプト**
   - 全体 & 法令別（`金商法` / `薬機法` / `借地借家法`）で  
     正解ラベル `a / b / c / d` の分布をカウントして表示する。

2. **薬機法 b/d 問題のスポットチェック用スクリプト**
   - `薬機法` かつ `correct_answer` が `b` or `d` のサンプルを数件抽出し、  
     ファイル名・正解ラベル・質問文・出力テキストを標準出力に表示する。

---

## 共通仕様

- 言語: **Python 3**
- 依存ライブラリ: **標準ライブラリのみ**
  - `json`
  - `collections.Counter`
  - `argparse`
  - 必要であれば `typing` 程度
- 文字コード: **UTF-8**
- JSONL 形式:
  - 1行1 JSON オブジェクト
- 対象ファイル:
  - デフォルトパス: `datasets/ft_direct_full_norag.jsonl`
  - コマンドライン引数でパスを変更できるようにすること（`--path` など）

### 想定する JSON 構造

`ft_direct_full_norag.jsonl` の 1行は、少なくとも次のキーを持っている想定で実装してください。

- `file_name`: 元の問題ファイル名（例: `"薬機法_第5章_選択式_関連法令_問題番号45"`）
- `correct_answer`: 正解ラベル（`"a"`, `"b"`, `"c"`, `"d"`のどれか）
- `question` または `input`: 質問文（どちらか一方が存在する想定）
- `output`: 教師データとして与える出力テキスト（モデルの理想回答）

**前提:**

- `correct_answer` が存在しない場合、その行はスキップしてください。
- `question` が存在しない場合は、代わりに `input` を使ってください。
- テキストを表示する際は、長くなりすぎないように一部のみ（先頭 N 文字）を切り出して表示してください。

---

## ファイル構成

1. `analyze_ft_label_distribution.py`
   - 役割: ラベル分布の集計（タスク 1）
2. `spotcheck_yakki_bd.py`
   - 役割: 薬機法 b/d 問題のスポットチェック（タスク 2）

別ファイルで作成してください。  
（もしサブコマンドでまとめたくなった場合は、1 ファイル内にサブコマンドを実装してもかまいませんが、  
ここでは「2 ファイルに分けて実装せよ」という前提で書いています）

---

## タスク 1: ラベル分布の集計スクリプト

ファイル名: `analyze_ft_label_distribution.py`

### 概要

- JSONL ファイルを 1行ずつ読み込み、
- `correct_answer` をカウントして、
  - **全体**
  - `file_name` が `金商法` で始まるサンプル
  - `file_name` が `薬機法` で始まるサンプル
  - `file_name` が `借地借家法` で始まるサンプル
- について、ラベル分布（`a, b, c, d` がそれぞれ何件か）を表示する。

### 詳細仕様

1. **コマンドライン引数**

   - `--path` または `-p`
     - 読み込む JSONL ファイルパス
     - デフォルト: `datasets/ft_direct_full_norag.jsonl`

   例:
   ```bash
   python analyze_ft_label_distribution.py \
     --path datasets/ft_direct_full_norag.jsonl
   ```

2. **内部処理**

   - `collections.Counter` を使ってカウントする。
   - 以下の Counter を用意すること:
     - `label_all`（全体）
     - `label_kinsyo`（`file_name` が `金商法` で始まる）
     - `label_yakki`（`file_name` が `薬機法` で始まる）
     - `label_shakuchi`（`file_name` が `借地借家法` で始まる）

   - ファイル読み込みループの流れ:
     1. 空行はスキップ
     2. `json.loads` でパース
     3. `file_name = obj.get("file_name", "")`
     4. `label = obj.get("correct_answer")`
     5. `label` が `None` の場合はスキップ
     6. `label_all[label] += 1`
     7. `file_name` の prefix を見て、該当する Counter にも加算
        - `file_name.startswith("金商法")` → `label_kinsyo[label] += 1`
        - `file_name.startswith("薬機法")` → `label_yakki[label] += 1`
        - `file_name.startswith("借地借家法")` → `label_shakuchi[label] += 1`

3. **出力フォーマット（標準出力）**

   - 人間が読みやすいテキスト形式で OK。
   - 例えば:

   ```text
   === 全体のラベル分布 ===
   a: 123
   b: 45
   c: 67
   d: 89

   === 金商法のラベル分布 ===
   a: 10
   b: 5
   c: 8
   d: 2

   === 薬機法のラベル分布 ===
   a: 3
   b: 12
   c: 1
   d: 9

   === 借地借家法のラベル分布 ===
   a: 4
   b: 3
   c: 0
   d: 7
   ```

   - ラベルが存在しない場合、そのラベルは表示されなくても構いませんが、  
     可能であれば `a, b, c, d` の順で 0 件でも表示すると見やすいです。

---

## タスク 2: 薬機法の b/d 問題スポットチェック

ファイル名: `spotcheck_yakki_bd.py`

### 概要

- JSONL ファイルから、
  - `file_name` が `薬機法` で始まる
  - `correct_answer` が `"b"` または `"d"`
- という条件を満たすサンプルを複数件抽出し、
- 次の情報を標準出力に表示する:
  - `file_name`
  - `correct_answer`
  - 質問文（`question` or `input`）
  - `output`（教師出力）

これにより、「薬機法の b/d 問題の教師ラベルやフォーマットに異常がないか」を目視確認できるようにします。

### 詳細仕様

1. **コマンドライン引数**

   - `--path` または `-p`
     - 読み込む JSONL ファイルパス
     - デフォルト: `datasets/ft_direct_full_norag.jsonl`
   - `--max-samples` または `-n`
     - 抽出して表示する最大件数
     - デフォルト: `10`

   例:
   ```bash
   python spotcheck_yakki_bd.py \
     --path datasets/ft_direct_full_norag.jsonl \
     --max-samples 10
   ```

2. **内部処理**

   - ファイルを 1行ずつ読み込む。
   - 空行はスキップ。
   - `json.loads` でパース。
   - 以下のように値を取得:
     - `file_name = obj.get("file_name", "")`
     - `label = obj.get("correct_answer")`
   - フィルタ条件:
     - `file_name.startswith("薬機法")` でない場合は continue
     - `label` が `"b"` でも `"d"` でもない場合は continue
   - 表示用に次の値を用意:
     - `question_text`
       - `obj.get("question")` があればそれを使用
       - なければ `obj.get("input", "")` を使用
       - 表示の際、長すぎる場合は先頭 120 文字程度に切り詰める。
     - `output_text`
       - `obj.get("output", "")` を使用
       - 表示の際、長すぎる場合は先頭 200 文字程度に切り詰める。

   - 見つけたサンプルはリストに追加し、`max_samples` 件に達したらループを終了する。

3. **出力フォーマット（標準出力）**

   - 人間が確認しやすい形式で、区切り線を入れて出力してください。
   - 例:

   ```text
   === 薬機法 b/d 問題のサンプル ===
   ----------------------------------------
   file_name : 薬機法_第5章_選択式_関連法令_問題番号45
   label    : b
   question : （ここに質問文の先頭 120 文字程度）
   output   : （ここに output の先頭 200 文字程度）

   ----------------------------------------
   file_name : 薬機法_第15章_選択式_関連法令_問題番号19
   label    : d
   question : （以下同様）
   output   : （以下同様）
   ```

   - 改行やインデントは多少ずれてもよいですが、  
     1 サンプルごとに区切り線を表示すること。

---

## 実装上の注意点

- JSONL が大きい可能性があるため、**必ずストリーム処理（1行ずつ処理）**を行い、  
  ファイル全体をメモリに読み込まないようにしてください。
- 例外処理:
  - JSON パースエラーが出る可能性がある場合は、  
    try/except で 1行単位のエラーを握りつぶすか、  
    エラー行をログ出力してスキップする実装だと親切です。
- 日本語ファイル名（`金商法` / `薬機法` / `借地借家法`）に対して  
  `startswith` を使うため、ファイルを開く際は `encoding="utf-8"` を指定してください。

---

## 期待する使い方イメージ

1. ラベル分布の確認:
   ```bash
   python analyze_ft_label_distribution.py \
     --path datasets/ft_direct_full_norag.jsonl
   ```

2. 薬機法 b/d 問題のスポットチェック:
   ```bash
   python spotcheck_yakki_bd.py \
     --path datasets/ft_direct_full_norag.jsonl \
     --max-samples 10
   ```

上記仕様を満たすように、2 つの Python スクリプトを生成してください。
