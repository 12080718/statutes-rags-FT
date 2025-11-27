# StepK 実行コマンド（RAGあり・v3評価のみ）

共通設定:
- モデル: `Qwen/Qwen3-8B`
- LoRA: `runs/qwen3_law_ft/direct_norag_q8_4bit_v3_b2s768`
- 4bit: `--hf-load-in-4bit`
- RAG: 有効（top-k 3）
- 評価のみ（FT v3）、dev/test 各30問想定（devは実際27件）

## dev（RAGあり・ベースラインFTなし）
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_dev.json \
  --output results/evaluations/qwen3_hf_rag_direct_dev30_v0.json \
  --samples 30 \
  --top-k 3 \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-load-in-4bit \
  --ensemble 1
```

## dev（RAGあり・FT v3）
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_dev.json \
  --output results/evaluations/qwen3_hf_ft_v3_rag_direct_dev30.json \
  --samples 30 \
  --top-k 3 \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-lora-path runs/qwen3_law_ft/direct_norag_q8_4bit_v3_b2s768 \
  --hf-load-in-4bit \
  --ensemble 1
```

## test（RAGあり・ベースラインFTなし）
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_test.json \
  --output results/evaluations/qwen3_hf_rag_direct_test30_v0.json \
  --samples 30 \
  --top-k 3 \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-load-in-4bit \
  --ensemble 1
```

## test（RAGあり・FT v3）
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_test.json \
  --output results/evaluations/qwen3_hf_ft_v3_rag_direct_test30.json \
  --samples 30 \
  --top-k 3 \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-lora-path runs/qwen3_law_ft/direct_norag_q8_4bit_v3_b2s768 \
  --hf-load-in-4bit \
  --ensemble 1
```
