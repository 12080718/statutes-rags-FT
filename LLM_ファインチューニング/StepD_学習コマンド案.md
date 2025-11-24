## StepD 学習コマンド案（実行しないでください）

前提: GPU 1枚（RTX3080クラス）を想定。バッチサイズは小さめ、LoRAで軽量化。モデル/データは適宜調整。

### 候補1: パターンA（no-RAG directモデル）
```
python scripts/train_qwen_law_ft.py \
  --model-name Qwen/Qwen1.5-7B-Chat \
  --train-file results/finetune/ft_direct_full_norag.jsonl \
  --output-dir runs/qwen_law_ft/direct_norag_v1 \
  --num-epochs 3 \
  --batch-size 2 \
  --learning-rate 2e-4 \
  --max-seq-length 1024 \
  --gradient-accumulation-steps 4 \
  --warmup-ratio 0.03 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --fp16
```

### 候補2: パターンB（no-RAG CoTモデル）
```
python scripts/train_qwen_law_ft.py \
  --model-name Qwen/Qwen1.5-7B-Chat \
  --train-file results/finetune/ft_cot_full_norag.jsonl \
  --output-dir runs/qwen_law_ft/cot_norag_v1 \
  --num-epochs 3 \
  --batch-size 2 \
  --learning-rate 2e-4 \
  --max-seq-length 1024 \
  --gradient-accumulation-steps 4 \
  --warmup-ratio 0.03 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --fp16
```

### 注意メモ
- GPUメモリに応じて `batch-size` と `gradient-accumulation-steps` を調整。厳しい場合は batch-size=1 + accum増加。
- `target-modules` はモデルに合わせて指定が必要な場合あり（例: `q_proj,v_proj` など）。現状は自動推測に任せる。
- 長文が多い場合は `max-seq-length` を短くし、Out of Memory を避ける。
- `bf16` 環境があれば `--bf16` を優先。
- 実行後は `trainer.train()` を有効化する必要あり（スクリプト内はコメントアウト状態）。
