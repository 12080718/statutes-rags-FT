# RAGあり/なし × FTあり/なし 実行コマンド（dev/test）

共通前提:
- HFバックエンド、4bit読み込み (`--hf-load-in-4bit`)
- モデル: `Qwen/Qwen3-8B`
- LoRAパス（FTあり時）: `runs/qwen3_law_ft/direct_v2_norag_q8_4bit_v1`
- dev: `datasets/lawqa_jp/data/selection_dev.json`（30問想定）
- test: `datasets/lawqa_jp/data/selection_test.json`（30問想定）

## RAGなし・FTなし
- dev
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_dev.json \
  --output results/evaluations/qwen3_hf_norag_direct_dev.json \
  --samples 30 \
  --top-k 1 \
  --no-rag \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-load-in-4bit \
  --ensemble 1
```
- test
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_test.json \
  --output results/evaluations/qwen3_hf_norag_direct_test.json \
  --samples 30 \
  --top-k 1 \
  --no-rag \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-load-in-4bit \
  --ensemble 1
```

## RAGなし・FTあり
- dev
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_dev.json \
  --output results/evaluations/qwen3_hf_ft_v2_norag_direct_dev.json \
  --samples 30 \
  --top-k 1 \
  --no-rag \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-lora-path runs/qwen3_law_ft/direct_v2_norag_q8_4bit_v1 \
  --hf-load-in-4bit \
  --ensemble 1
```
- test
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_test.json \
  --output results/evaluations/qwen3_hf_ft_v2_norag_direct_test.json \
  --samples 30 \
  --top-k 1 \
  --no-rag \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-lora-path runs/qwen3_law_ft/direct_v2_norag_q8_4bit_v1 \
  --hf-load-in-4bit \
  --ensemble 1
```

## RAGあり・FTなし
- dev
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_dev.json \
  --output results/evaluations/qwen3_hf_rag_direct_dev.json \
  --samples 30 \
  --top-k 3 \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-load-in-4bit \
  --ensemble 1
```
- test
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_test.json \
  --output results/evaluations/qwen3_hf_rag_direct_test.json \
  --samples 30 \
  --top-k 3 \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-load-in-4bit \
  --ensemble 1
```

## RAGあり・FTあり
- dev
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_dev.json \
  --output results/evaluations/qwen3_hf_ft_v2_rag_direct_dev.json \
  --samples 30 \
  --top-k 3 \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-lora-path runs/qwen3_law_ft/direct_v2_norag_q8_4bit_v1 \
  --hf-load-in-4bit \
  --ensemble 1
```
- test
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_test.json \
  --output results/evaluations/qwen3_hf_ft_v2_rag_direct_test.json \
  --samples 30 \
  --top-k 3 \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-lora-path runs/qwen3_law_ft/direct_v2_norag_q8_4bit_v1 \
  --hf-load-in-4bit \
  --ensemble 1
```

## RAGあり（hybrid）に切り替える場合
- コマンド先頭に `RETRIEVER_TYPE=hybrid` を付与（RAGありパターンのみ）。
- RAGあり・FTなし dev
```
RETRIEVER_TYPE=hybrid python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_dev.json \
  --output results/evaluations/qwen3_hf_rag_direct_dev_hybrid.json \
  --samples 30 \
  --top-k 3 \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-load-in-4bit \
  --ensemble 1
```
- RAGあり・FTなし test
```
RETRIEVER_TYPE=hybrid python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_test.json \
  --output results/evaluations/qwen3_hf_rag_direct_test_hybrid.json \
  --samples 30 \
  --top-k 3 \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-load-in-4bit \
  --ensemble 1
```
- RAGあり・FTあり dev
```
RETRIEVER_TYPE=hybrid python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_dev.json \
  --output results/evaluations/qwen3_hf_ft_v2_rag_direct_dev_hybrid.json \
  --samples 30 \
  --top-k 3 \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-lora-path runs/qwen3_law_ft/direct_v2_norag_q8_4bit_v1 \
  --hf-load-in-4bit \
  --ensemble 1
```
- RAGあり・FTあり test
```
RETRIEVER_TYPE=hybrid python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_test.json \
  --output results/evaluations/qwen3_hf_ft_v2_rag_direct_test_hybrid.json \
  --samples 30 \
  --top-k 3 \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-lora-path runs/qwen3_law_ft/direct_v2_norag_q8_4bit_v1 \
  --hf-load-in-4bit \
  --ensemble 1
```
