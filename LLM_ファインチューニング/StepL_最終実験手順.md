# StepL 最終実験手順（no-RAG 直答最終評価: train100/test40, v4想定）

## 1. selection 最終split（train100/test40）
- スクリプト: `scripts/split_lawqa_selection_final.py`
- 実行例（seed=20251128固定）:
```
python scripts/split_lawqa_selection_final.py
```
- 出力:  
  - `datasets/lawqa_jp/data/selection_train_final100.json`  
  - `datasets/lawqa_jp/data/selection_test_final40.json`  
  分布（law_name/answer）がコンソールに出るので偏りを目視確認。

## 2. v4用JSONL生成（no-RAG direct, train100/dev27）
- スクリプト: `scripts/build_finetune_dataset.py`
```
python scripts/build_finetune_dataset.py \
  --lawqa-path datasets/lawqa_jp/data/selection_train_final100.json \
  --output-path results/finetune/ft_direct_v4_train_norag.jsonl \
  --mode direct --no-rag --few-shot

python scripts/build_finetune_dataset.py \
  --lawqa-path datasets/lawqa_jp/data/selection_dev.json \
  --output-path results/finetune/ft_direct_v4_dev_norag.jsonl \
  --mode direct --no-rag --few-shot
```
※ test40は評価専用。devは現行27問を使用。

## 3. 学習（no-RAG 直答 v4, A100想定）
- スクリプト: `scripts/train_qwen_law_ft.py`
```
python scripts/train_qwen_law_ft.py \
  --model-name "Qwen/Qwen3-8B" \
  --train-file results/finetune/ft_direct_v4_train_norag.jsonl \
  --output-dir runs/qwen3_law_ft/direct_norag_q8_4bit_v4 \
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
※ メモリに応じて batch/seq を調整可。出力先は適宜命名。

## 4. 評価（no-RAGのみ, dev27/test40, パーサv3）
- スクリプト: `scripts/evaluate_multiple_choice.py`
- ベースライン（FTなし）:
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_dev.json \
  --output results/evaluations/qwen3_hf_norag_direct_dev_v0.json \
  --samples 30 \
  --top-k 1 \
  --no-rag \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-load-in-4bit \
  --ensemble 1

python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_test_final40.json \
  --output results/evaluations/qwen3_hf_norag_direct_test40_v0.json \
  --samples 40 \
  --top-k 1 \
  --no-rag \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-load-in-4bit \
  --ensemble 1
```
- FT v4:
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_dev.json \
  --output results/evaluations/qwen3_hf_ft_v4_norag_direct_dev.json \
  --samples 30 \
  --top-k 1 \
  --no-rag \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-lora-path runs/qwen3_law_ft/direct_norag_q8_4bit_v4 \
  --hf-load-in-4bit \
  --ensemble 1

python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection_test_final40.json \
  --output results/evaluations/qwen3_hf_ft_v4_norag_direct_test40.json \
  --samples 40 \
  --top-k 1 \
  --no-rag \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-lora-path runs/qwen3_law_ft/direct_norag_q8_4bit_v4 \
  --hf-load-in-4bit \
  --ensemble 1
```
※ 日時プレフィックス運用なら適宜付与。パーサはv3（Answer/回答行優先）を使用。

## 5. 確認ポイント
- train/test件数: 100/40 で固定されていること。
- 学習完走とLoRA出力の存在（指定パス）。
- dev/testでFT v4がベースラインを上回るか（特にtest40が最終指標）。
- unknown/パースエラー件数、ラベル分布の偏りを確認。***
