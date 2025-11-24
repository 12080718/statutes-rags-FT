## StepE-4 評価コマンド案（HFバックエンド前提、実行しないでください）

前提: HF+LoRAモデルを `--llm-backend hf` で呼び出す設計。RAG top_k=3、デフォルト retriever/reranker設定。モデル/LoRAパスは適宜置換。

- 共通オプション例
  - `--llm-backend hf`
  - `--hf-model-name Qwen/Qwen1.5-7B-Chat`
  - `--hf-lora-path runs/qwen_law_ft/direct_norag_v1`（FTのみ）
  - `--top-k 3`（RAGあり）

### Base HFモデル（LoRAなし）
1) RAGあり + direct
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection.json \
  --output results/evaluations/hf_base_rag_direct.json \
  --top-k 3 \
  --llm-backend hf \
  --hf-model-name Qwen/Qwen1.5-7B-Chat \
  --ensemble 1
```

2) RAGあり + CoT
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection.json \
  --output results/evaluations/hf_base_rag_cot.json \
  --top-k 3 \
  --llm-backend hf \
  --hf-model-name Qwen/Qwen1.5-7B-Chat \
  --ensemble 1 \
  --use-cot
```

### FT HFモデル（LoRA適用）
3) RAGあり + direct
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection.json \
  --output results/evaluations/hf_ft_rag_direct.json \
  --top-k 3 \
  --llm-backend hf \
  --hf-model-name Qwen/Qwen1.5-7B-Chat \
  --hf-lora-path runs/qwen_law_ft/direct_norag_v1 \
  --ensemble 1
```

4) RAGあり + CoT
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection.json \
  --output results/evaluations/hf_ft_rag_cot.json \
  --top-k 3 \
  --llm-backend hf \
  --hf-model-name Qwen/Qwen1.5-7B-Chat \
  --hf-lora-path runs/qwen_law_ft/cot_norag_v1 \
  --ensemble 1 \
  --use-cot
```

### メモ
- Rerankerを併用する場合は `--use-reranker --rerank-top-n 3` などを追加。
- Ollamaバックエンドとの比較をする場合は、同条件で `--llm-backend ollama --llm-model qwen3:8b` のコマンドを用意するとよい。
