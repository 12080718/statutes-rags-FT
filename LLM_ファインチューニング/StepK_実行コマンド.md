# StepK 実行コマンド（no-RAG 直答 v3）

## v3 用 JSONL 生成
- train (83件)
```
python scripts/build_finetune_dataset.py \
  --lawqa-path datasets/lawqa_jp/data/selection_train.json \
  --output-path results/finetune/ft_direct_v3_train_norag.jsonl \
  --mode direct \
  --no-rag \
  --few-shot
```

## v3 学習（no-RAG 直答 v3, A100想定）
```
python scripts/train_qwen_law_ft.py \
  --model-name "Qwen/Qwen3-8B" \
  --train-file results/finetune/ft_direct_v3_train_norag.jsonl \
  --output-dir runs/qwen3_law_ft/direct_norag_q8_4bit_v3 \
  --num-epochs 1 \
  --batch-size 8 \
  --max-seq-length 1024 \
  --learning-rate 2e-5 \
  --gradient-accumulation-steps 4 \
  --warmup-ratio 0.1 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --use-4bit \
  --bnb-4bit-compute-dtype bfloat16 \
  --bnb-4bit-quant-type nf4 \
  --do-train
```

## v3 評価（dev/test, ベースライン vs FT v3）
- dev 30問 ベースライン（FTなし）
```
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
- dev 30問 FT v3
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_dev.json \
  --output results/evaluations/qwen3_hf_ft_norag_direct_dev30_v3.json \
  --samples 30 \
  --top-k 1 \
  --no-rag \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-lora-path runs/qwen3_law_ft/direct_norag_q8_4bit_v3 \
  --hf-load-in-4bit \
  --ensemble 1
```
- test 30問 ベースライン（FTなし）
```
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
- test 30問 FT v3
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_test.json \
  --output results/evaluations/qwen3_hf_ft_norag_direct_test30_v3.json \
  --samples 30 \
  --top-k 1 \
  --no-rag \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-lora-path runs/qwen3_law_ft/direct_norag_q8_4bit_v3 \
  --hf-load-in-4bit \
  --ensemble 1
```
- dev (27件)
```
python scripts/build_finetune_dataset.py \
  --lawqa-path datasets/lawqa_jp/data/selection_dev.json \
  --output-path results/finetune/ft_direct_v3_dev_norag.jsonl \
  --mode direct \
  --no-rag \
  --few-shot
```
