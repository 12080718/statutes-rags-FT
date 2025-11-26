# A100 側 no-RAG 直答実験の結果共有と今後の方針（3080/Codex 向けメモ）

このファイルは、heart01(A100) で実行した no-RAG 直答実験の結果と、
FT(ファインチューニング)ありで精度が悪化した原因整理、
および今後 3080 側で進めたい方針を共有するためのメモです。

---

## 1. 実験条件（共通）

- データセット: `lawqa_jp` の `selection.json`（全 140 問）
- タスク: no-RAG 4択直答
  - few-shot: 有効
  - CoT: 無効
  - RAG: 無効 (`--no-rag`)
  - retriever: `bm25`（ただし no-RAG なので実質未使用）
  - reranker: なし
  - top-k: 1
- 評価スクリプト:
  - `scripts/evaluate_multiple_choice.py`
- LLM バックエンド:
  - `--llm-backend hf`
  - `--hf-model-name "Qwen/Qwen3-8B"`
  - `--hf-load-in-4bit`

---

## 2. A100 側の結果まとめ

### 2-1. RAG なし・FT なし（ベースライン）

- 設定: HF Qwen3-8B そのまま（LoRA なし）
- 結果（selection 140 問トータル）
  - 正答: **60 / 140**
  - 正答率: **42.9 %**
  - `unknown`（パース失敗）: 3 件
- コメント:
  - 4択のランダム(25%)を大きく上回っており、
    「**素の Qwen3-8B + few-shot**」としてはそこそこ強いベースライン。

### 2-2. RAG なし・FT あり（v1: direct_norag_q8_4bit_v1）

- 設定:
  - モデル: Qwen/Qwen3-8B
  - 4bit QLoRA
  - 学習データ: `ft_direct_full_norag.jsonl`（**selection 140 問をそのまま全部使った JSONL**）
  - 学習エポック: 3 epoch（v1）
- 結果（selection 140 問トータル）
  - 正答: **39 / 140**
  - 正答率: **27.9 %**
  - `unknown`（パース失敗）: 18 件
- no-FT ベースラインとの比較:
  - 正答数: 60 → 39（▲21 問）
  - 正答率: 42.9 % → 27.9 %（▲15 pt）
  - `unknown`: 3 → 18（+15 件）
- つまり、**v1 の FT は「学習に同じ 140 問を使っているにもかかわらず、むしろ精度を悪化させている」**。

---

## 3. FT ありで結果が悪くなった主な原因

heart01 側で JSON を突き合わせた結果、主な原因は以下の 3 点に集約されます。

### 3-1. 学習データと評価データが完全に同じ（selection 140 問を丸ごと学習）

- `ft_direct_full_norag.jsonl` は selection の 140 問をそのまま全件使用。
- 本来であれば「train に使った問題では 100% 正解に近づく」ことが多いが、
  実際には 27.9% まで悪化している。
- これは、
  - **データ数の少なさ（140 問だけ）**
  - **LoRA のエポック数・学習率などハイパラ**
  により、ベースモデルの挙動を「変な方向に上書きしてしまった」可能性が高い。

### 3-2. 出力フォーマットとパーサのミスマッチによる `unknown` 激増

- 評価スクリプト側は、基本的に **半角 `a/b/c/d` 1 文字だけ** を正解候補として想定している。
- 一方、FT 後の Qwen 出力は、

  - `ｂ
回答（1文字のみ）: ｂ
回答（1文字のみ）: ｂ
...`
  - `4
回答（1文字のみ）: 4
回答（1文字のみ）: 4
...`

  のように、
  - **全角 1 文字** (`ｂ` など)
  - **数字 1〜4**
  - 「回答（1文字のみ）:」などの指示文をオウム返ししながらループ
  する出力が多くなっている。

- 現行パーサはこれらをすべて弾いてしまうため、`unknown` が **3 件 → 18 件** に増加している。
- 全角 → 半角、数字 → a〜d 変換で救済すると、18 件中 7 件程度は「本当は正解」になり得る。

### 3-3. LoRA 学習により「とりあえず a を出す」挙動に崩壊

- 予測ラベル分布（140 問）を比較すると：

  - no-FT: a 66, b 30, c 21, d 20, unknown 3
  - FT v1: a **100**, b 7, c 5, d 10, unknown 18

- 一方、正解ラベルの分布は
  - a 23, b 37, c 48, d 32
  であり、本来は **c が一番多い**。
- それにもかかわらず、FT 後は出力が **a に極端に偏っている**。
- 実際、
  - 「元は b/c/d で正解していたのに、FT 後は a に変わって不正解」
  というケースが多数確認された。
- 推測される要因：

  - 学習データが少ない状態で、
  - 指示文を含むプロンプトをそのまま LoRA にかけた結果、
    **「内容理解」よりも「形式（回答: a 等）の模倣」に過剰適応した**。

---

## 4. 今後の方針（3080/Codex に共有したいこと）

卒研としての「今やるべきこと（ゴール）」は以下の 3 点です。

1. `lawqa_jp` の `selection.json` を **train/dev/test** に分ける  
   - 例: train 80 / dev 30 / test 30（科目ごとに比率を維持）。
2. train 80 問だけを使って **FT v2（no-RAG 直答 LoRA）** を実施する。  
3. その FT v2 モデルで、**no-RAG 直答のベースライン 42.9% を「確実に」超える**。

この方針に沿って、3080/Codex 側では主に次のような作業をお願いしたいです。

### 4-1. answer パーサの改善（評価ロジック修正）

- 対象: `scripts/evaluate_multiple_choice.py`
- 目的:
  - 全角 → 半角正規化、数字 1〜4 → a〜d 変換、
    `回答: X` / `Answer: X` 行の優先パースを行う関数
    `normalize_and_parse_answer()` を追加して、現行のパース処理を置き換える。
- 別途渡している `evaluate_answer_parser_fix_instructions.md` の内容に従って改修をお願いしたいです。

### 4-2. selection の train/dev/test 分割

- `datasets/lawqa_jp/data/selection.json` を、
  - `selection_train.json` (80 問)
  - `selection_dev.json` (30 問)
  - `selection_test.json` (30 問)
  に分割するスクリプトを作成してもらいたいです。
- 理想は科目（金商法 / 薬機法 / 借地借家法）ごとに 60/20/20 くらいの比率を維持する stratified split。
- 3080 側で `scripts/split_selection_dataset.py` を追加（デフォルト 60/20/20, seed=42）。実行例:
  ```
  python scripts/split_selection_dataset.py \
    --input datasets/lawqa_jp/data/selection.json \
    --output-train datasets/lawqa_jp/data/selection_train.json \
    --output-dev datasets/lawqa_jp/data/selection_dev.json \
    --output-test datasets/lawqa_jp/data/selection_test.json \
    --train-ratio 0.6 \
    --dev-ratio 0.2 \
    --seed 42
  ```
  （train/dev 比率は引数で変更可。合計が1.0以下になるように設定）

### 4-3. FT v2 用 JSONL の生成と評価スクリプトの実行

- A100 側での学習に向けて、次のコマンドで JSONL を作成しておきたいです。

```bash
python scripts/build_finetune_dataset.py   --lawqa-path datasets/lawqa_jp/data/selection_train.json   --output-path results/finetune/ft_direct_v2_train_norag.jsonl   --mode direct   --no-rag   --few-shot

python scripts/build_finetune_dataset.py   --lawqa-path datasets/lawqa_jp/data/selection_dev.json   --output-path results/finetune/ft_direct_v2_dev_norag.jsonl   --mode direct   --no-rag   --few-shot
```

- 学習後は、3080 側で以下の評価コマンドを使って、
  - FT なしベースライン v0
  - FT v2（LoRA あり）
  を dev/test それぞれで比較する予定です。

- LoRA v2 学習コマンド例（A100, no-RAG direct, 4bit QLoRA, train=selection_train.json）:

```bash
python scripts/train_qwen_law_ft.py \
  --model-name "Qwen/Qwen3-8B" \
  --train-file results/finetune/ft_direct_v2_train_norag.jsonl \
  --output-dir runs/qwen3_law_ft/direct_v2_norag_q8_4bit_v1 \
  --num-epochs 3 \
  --batch-size 2 \
  --learning-rate 2e-4 \
  --max-seq-length 1024 \
  --gradient-accumulation-steps 4 \
  --warmup-ratio 0.03 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --use-4bit \
  --bnb-4bit-compute-dtype bfloat16 \
  --bnb-4bit-quant-type nf4 \
  --do-train
```
- 短時間の動作確認（50ステップなど）を先に行う場合は、`--max-steps-override 50` や `--num-epochs 1 --batch-size 1` に一時的に下げてから本番設定に戻す。

（評価コマンド例は別途 StepD テンプレの md で共有済みです）

---

## 5. まとめ

- heart01(A100) での v1 実験では、
  - **RAGなし・FTなし**: 42.9% (60/140)
  - **RAGなし・FTあり (v1)**: 27.9% (39/140)
- FT ありで悪化した主因は、
  - selection 140 問をそのまま全件学習に使ったことによる「変な方向への上書き」
  - answer パーサが厳しすぎることによる `unknown` 激増
  - LoRA 学習により「とりあえず a を出す」挙動に崩壊したこと
- 今後は、
  - selection を train/dev/test に分割し、
  - train80 のみで FT v2 を行い、
  - 改善された answer パーサのもとで dev/test の精度を評価する、
  という流れで「no-RAG 直答ベースライン 42.9% を確実に超える」ことを目標とする。

以上を 3080 側（Codex）に共有しておきたいです。
