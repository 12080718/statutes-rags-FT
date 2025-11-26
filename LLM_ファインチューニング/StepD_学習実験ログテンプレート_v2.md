## 実験ID: Exp-v2-norag-direct-q8-4bit

### 1. 目的
- lawqa_jp selection を train/dev/test に分割した上で、
  - no-RAG直答のベースライン (Qwen3-8B, FTなし)
  - QLoRA FT v2 (direct_norag_q8_4bit_v2)
  を dev/test で比較し、ベースライン 42.9% を上回る設定を探す。

### 2. 使用データ
- データセット: lawqa_jp
- スプリット:
  - train: selection_train.json (80問)
  - dev:   selection_dev.json (30問)
  - test:  selection_test.json (30問)
- 前処理:
  - build_finetune_dataset.py で JSONL 生成
    - train: results/finetune/ft_direct_v2_train_norag.jsonl
    - dev:   results/finetune/ft_direct_v2_dev_norag.jsonl

### 3. モデル・環境
- ベースモデル: Qwen/Qwen3-8B
- LoRA: 4bit QLoRA, r=32, alpha=64, dropout=0.05
- max_seq_length: 1024
- 学習環境: heart01 (A100 80GB) など

### 4. 実行コマンド

#### 4-1. 学習 (train_qwen_law_ft.py)

```bash
python scripts/train_qwen_law_ft.py \
  --model-name "Qwen/Qwen3-8B" \
  --train-file results/finetune/ft_direct_v2_train_norag.jsonl \
  --output-dir runs/qwen3_law_ft/direct_norag_q8_4bit_v2 \
  --num-epochs 2 \
  --batch-size 8 \
  --max-seq-length 1024 \
  --learning-rate 5e-5 \
  --gradient-accumulation-steps 4 \
  --warmup-ratio 0.1 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --use-4bit \
  --bnb-4bit-compute-dtype bfloat16 \
  --bnb-4bit-quant-type nf4
```

#### 4-2. 評価 (dev/test, no-FT baseline & FT v2)

- dev 30問 (no-FT baseline):

```bash
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_dev.json \
  --output results/evaluations/qwen3_hf_norag_direct_dev30_v0.json \
  --samples 30 \
  --top-k 1 \
  --no-rag \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-load-in-4bit \
  --ensemble 1
```

- dev 30問 (FT v2):

```bash
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_dev.json \
  --output results/evaluations/qwen3_hf_ft_norag_direct_dev30_v2.json \
  --samples 30 \
  --top-k 1 \
  --no-rag \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-lora-path runs/qwen3_law_ft/direct_norag_q8_4bit_v2 \
  --hf-load-in-4bit \
  --ensemble 1
```

- test 30問 (no-FT baseline):

```bash
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_test.json \
  --output results/evaluations/qwen3_hf_norag_direct_test30_v0.json \
  --samples 30 \
  --top-k 1 \
  --no-rag \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-load-in-4bit \
  --ensemble 1
```

- test 30問 (FT v2):

```bash
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_test.json \
  --output results/evaluations/qwen3_hf_ft_norag_direct_test30_v2.json \
  --samples 30 \
  --top-k 1 \
  --no-rag \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-lora-path runs/qwen3_law_ft/direct_norag_q8_4bit_v2 \
  --hf-load-in-4bit \
  --ensemble 1
```

### 5. 結果サマリ

#### 5-1. dev 30問

| モデル               | 正答/全体 | 正答率 | unknown数 | 備考 |
|----------------------|-----------|--------|-----------|------|
| v0: no-FT baseline   | xx / 30   | xx.x%  | u0        |      |
| v2: LoRA FT (Exp-v2) | yy / 30   | yy.y%  | u1        |      |

#### 5-2. test 30問

| モデル               | 正答/全体 | 正答率 | unknown数 | 備考             |
|----------------------|-----------|--------|-----------|------------------|
| v0: no-FT baseline   | aa / 30   | aa.a%  | t0        |                  |
| v2: LoRA FT (Exp-v2) | bb / 30   | bb.b%  | t1        | ★ベースライン超え？ |

### 6. 考察（メモ用）
- dev/test の両方で LoRA v2 がベースラインを上回ったか？
- v1 で問題だった「a に偏る」「unknown 激増」は改善したか？
- 科目別（金商法/薬機法/借地借家法）の精度の差はどうか？

### 7. 次アクション
- test30 で十分な改善が見られたら：
  - full140 での参考評価を実施
  - RAGあり版への展開を検討
- 改善が弱い場合：
  - epoch数や learning rate の再調整案を検討
