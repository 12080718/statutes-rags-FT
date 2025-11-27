# statutes-rags-FT: no-RAG 直答 v3 用プロンプト設計 & 学習設定 変更指示（Codex 向け）

このファイルは、Qwen3-8B の **no-RAG 直答モデル v3** を作るために、
プロンプト設計と学習設定を調整してもらうための指示です。

目的：
- v1 / v2 で発生していた
  - 「a に過度に寄る出力」
  - 「回答（1文字のみ）: ...」のオウム返し
- を抑えつつ、
  - **no-RAG ベースライン 42.9% を安定して上回る** モデルを目指します。

---

## 1. 変更方針の全体像

1. **FT 用プロンプト（学習データの input）から、過剰なフォーマット指示を削る**
   - 特に `回答（1文字のみ）: ` などのフレーズを **学習データからは除去** する。
2. **出力フォーマットを「Answer: a」のようなシンプルな形に統一**
   - 学習時の `output` を `Answer: a` 形式にして、
   - 評価時は既に実装済みの `normalize_and_parse_answer()` で `a` を抽出する。
3. **LoRA の「強さ」を更に弱める（epoch 1 + 低 learning rate + 低 r）**
   - ベースモデルの挙動を壊さない方向に寄せる。

---

## 2. 影響範囲

### 2-1. プロンプト生成

- 対象ファイル:  
  - `scripts/build_finetune_dataset.py`

- 対象モード:  
  - `--mode direct` かつ `--no-rag` のときに生成される `input` / `output`

### 2-2. 学習設定（LoRA）

- 対象ファイル:
  - `scripts/train_qwen_law_ft.py`（コード自体はそのままでも良い）
- 対象コマンド:
  - no-RAG 直答 v3 用の実験コマンド（下に記載）を新たに用意

---

## 3. プロンプト設計の変更（v3）

### 3-1. 現状イメージ（v1/v2 の学習プロンプト）

`build_finetune_dataset.py` の `input` は、ざっくり下記のような構造になっている想定です：

- 冒頭で「あなたは日本法のアシスタントです…」
- 問題文＋選択肢
- 「a, b, c, d から1つ選んでください」
- 末尾に `回答（1文字のみ）:` のような行

この **「回答（1文字のみ）:」という固定フレーズを含んだテンプレ** が、
LoRA 学習によってオウム返しされ、`a` へのバイアスを強めている疑いがあります。

### 3-2. v3 での新しいプロンプト仕様

`build_finetune_dataset.py` の direct/no-rag 用テンプレートを、**以下の方針で書き換えてください。**

#### (A) input テンプレート（学習用プロンプト）

- 例：Python 風に書くと以下のような文字列になるイメージです。  
  （実際の変数名は既存コードに合わせてください）

```python
prompt = (
    "あなたは日本の法律に関する4択問題に答えるアシスタントです。\n"
    "次の問題文と選択肢を読み、最も適切なものを1つ選んでください。\n\n"
    "問題:\n"
    f"{question}\n\n"
    "選択肢:\n"
    f"a. {option_a}\n"
    f"b. {option_b}\n"
    f"c. {option_c}\n"
    f"d. {option_d}\n\n"
    "法律の知識に基づき、正しい選択肢を一つだけ選びなさい。\n"
    "最終行で、次の形式で答えてください。\n\n"
    "Answer: a\n"
)
```

**重要ポイント：**

- `回答（1文字のみ）:` や「1文字だけを出力せよ」といった文言は **入れない**。
- 「Answer: a」の例は **1回だけ** 示す。
- 実際の解答は **output 側** で指定する（下を参照）。

#### (B) output テンプレート

- output は **必ず 1 行で、`Answer: {correct_letter}`** という形式にしてください。

```python
output = f"Answer: {correct_answer_letter}"
```

- `correct_answer_letter` は `a` / `b` / `c` / `d` のいずれか。
- これにより、学習時は常に
  - `input`: 上記の自然言語プロンプト
  - `output`: `Answer: b`
  というペアでLoRA学習が行われます。

#### (C) 実装上のメモ

- `build_finetune_dataset.py` 内で、
  - 既存の direct/no-rag 用プロンプト生成ロジックを探し、
  - `回答（1文字のみ）:` を含む部分を **この新テンプレに置き換える** 形で修正してください。
- 目印となる文字列候補：
  - `"回答（1文字のみ）"` や `"1文字のみ"`
  - `"no-rag"` / `"direct"` モード分岐周辺

- 出力 JSONL については、v3 用に **別ファイル名** にします（次項）。

---

## 4. v3 用 JSONL 生成コマンド

プロンプト仕様変更後、以下のコマンドで **新しい v3 用 JSONL** を生成してください。

### 4-1. train 用（80問）

```bash
python scripts/build_finetune_dataset.py   --lawqa-path datasets/lawqa_jp/data/selection_train.json   --output-path results/finetune/ft_direct_v3_train_norag.jsonl   --mode direct   --no-rag   --few-shot
```

### 4-2. dev 用（30問）

```bash
python scripts/build_finetune_dataset.py   --lawqa-path datasets/lawqa_jp/data/selection_dev.json   --output-path results/finetune/ft_direct_v3_dev_norag.jsonl   --mode direct   --no-rag   --few-shot
```

- `selection_train/dev.json` は、既に作成済みのもの（train80 / dev30）を前提とします。
- v2 用の JSONL（`ft_direct_v2_*`）は残したままで構いません。

---

## 5. 学習設定（v3 実験用）の提案

心臓部（heart01/A100）で実行する学習コマンド例です。  
コード側の変更は不要で、**コマンドのハイパラだけ調整**すればOKです。

### 5-1. v3 学習コマンド（no-RAG 直答 v3）

```bash
python scripts/train_qwen_law_ft.py \
  --model-name "Qwen/Qwen3-8B" \
  --train-file results/finetune/ft_direct_v3_train_norag.jsonl \
  --output-dir runs/qwen3_law_ft/direct_norag_q8_4bit_v3_b2s768 \
  --num-epochs 1 \
  --batch-size 2 \
  --max-seq-length 768 \
  --learning-rate 2e-5 \
  --gradient-accumulation-steps 8 \
  --warmup-ratio 0.1 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --use-4bit \
  --bnb-4bit-compute-dtype bfloat16 \
  --bnb-4bit-quant-type nf4 \
  --do-train
```

#### ハイパラ変更の意図

- `num-epochs: 1`
  - 80問という小さなデータに対して過学習しすぎないようにする。
- `learning-rate: 2e-5`
  - v2 より小さくし、ベースモデルの決定境界を壊しにくくする。
- `lora-r: 16`, `lora-alpha: 32`
  - LoRA の表現力を少し弱めて、「フォーマットへの過適合」よりも
    軽いドメイン適応寄りに振る。
- その他の設定は v2 と近いまま。

---

## 6. 評価コマンド（参考）

評価スクリプト側では、すでに `normalize_and_parse_answer()` を導入済みの前提です。  
その上で、v3 については以下のように評価する想定です。

### 6-1. dev 30問（ベースライン vs v3）

```bash
# ベースライン（FTなし）
python scripts/evaluate_multiple_choice.py   --data datasets/lawqa_jp/data/selection_dev.json   --output results/evaluations/qwen3_hf_norag_direct_dev30_v0.json   --samples 30   --top-k 1   --no-rag   --llm-backend hf   --hf-model-name "Qwen/Qwen3-8B"   --hf-load-in-4bit   --ensemble 1

# FT v3
python scripts/evaluate_multiple_choice.py   --data datasets/lawqa_jp/data/selection_dev.json   --output results/evaluations/qwen3_hf_ft_norag_direct_dev30_v3.json   --samples 30   --top-k 1   --no-rag   --llm-backend hf   --hf-model-name "Qwen/Qwen3-8B"   --hf-lora-path runs/qwen3_law_ft/direct_norag_q8_4bit_v3   --hf-load-in-4bit   --ensemble 1
```

### 6-2. test 30問（最終確認）

```bash
# ベースライン（FTなし）
python scripts/evaluate_multiple_choice.py   --data datasets/lawqa_jp/data/selection_test.json   --output results/evaluations/qwen3_hf_norag_direct_test30_v0.json   --samples 30   --top-k 1   --no-rag   --llm-backend hf   --hf-model-name "Qwen/Qwen3-8B"   --hf-load-in-4bit   --ensemble 1

# FT v3
python scripts/evaluate_multiple_choice.py   --data datasets/lawqa_jp/data/selection_test.json   --output results/evaluations/qwen3_hf_ft_norag_direct_test30_v3.json   --samples 30   --top-k 1   --no-rag   --llm-backend hf   --hf-model-name "Qwen/Qwen3-8B"   --hf-lora-path runs/qwen3_law_ft/direct_norag_q8_4bit_v3   --hf-load-in-4bit   --ensemble 1
```

---

## 7. 期待する確認ポイント

- dev / test の両方で、
  - **FT v3 がベースライン v0（FTなし）を上回っているか？**
  - 特に **薬機法 b/d 問題** での精度が v2 より改善しているか？
- 予測ラベル分布（a/b/c/d）を集計し、
  - `a` に過度に偏っていないか？
  - `unknown` が異常に増えていないか？
  をチェックしてもらえると助かります。

---

以上の内容を前提に、

- `build_finetune_dataset.py` の direct/no-rag プロンプト生成の修正
- v3 用 JSONL 生成（train/dev）
- 学習コマンド & 評価コマンドのセットアップ

までを Codex 側で対応してもらえると嬉しいです。
