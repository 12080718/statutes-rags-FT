# StepI A100 Qwen3:8B 4bit QLoRA 本番学習指示（heart01用）

## 2-1. 目的
- heart01（A100 20GB/MIG）で Hugging Face バックエンドの Qwen3:8B 系モデル（ここでは `Qwen/Qwen3-8B` を利用）に対し、no-RAG direct の 4bit QLoRA 学習を実行し、lawqa_jp 140問の精度向上を狙う。
- 学習は HF 専用（`--llm-backend hf` 前提）、curl/Ollama は利用しない。Codex はローカル側での編集支援のみを担い、本指示内のコマンドは heart01 でユーザが実行する。

## 2-2. 前提環境の確認（heart01）
- GPU: `nvidia-smi` で A100 20GB (MIG スライス) を確認。
- 仮想環境: `.venv` を作成済みで `pip install -r requirements-llm.txt` を完了していること。
- データ: `datasets/lawqa_jp/data/selection.json` と `results/finetune/ft_direct_full_norag.jsonl` が存在すること。
- キャッシュ: `HF_HOME` を共有パス（例: `/home/jovyan/work/.cache/huggingface`）に設定済みであること。
- 本手順は A100 専用設定。RTX3080 では使用しない。

## 2-3. トライアル学習（50ステップ程度の動作確認）
- 目的: `Qwen/Qwen3-8B` のロード、4bit QLoRA 設定、データパスが正しく動くかを短時間で確認する。
- コマンド例（心配なら `batch-size 1` から開始）:
```
python scripts/train_qwen_law_ft.py \
  --model-name "Qwen/Qwen3-8B" \
  --train-file results/finetune/ft_direct_full_norag.jsonl \
  --output-dir runs/qwen3_law_ft/direct_norag_q8_4bit_trial \
  --num-epochs 1 \
  --batch-size 1 \
  --learning-rate 2e-4 \
  --max-seq-length 512 \
  --gradient-accumulation-steps 4 \
  --warmup-ratio 0.03 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --use-4bit \
  --bnb-4bit-compute-dtype bfloat16 \
  --bnb-4bit-quant-type nf4 \
  --max-steps-override 50 \
  --do-train
```
- 確認ポイント: 学習開始までにエラーがないか、1ステップあたりの所要時間、loss の減少傾向、GPUメモリ使用量。結果メモは `LLM_ファインチューニング/StepI_タスク2_結果.md` などに残す。

## 2-4. 本番学習（3エポック）
- 設定は `StepI_Qwen3_8B_設定案.md` を採用し、A100 20GB を想定。必要に応じて heart01 上で batch/seq を調整する。
- コマンド例:
```
python scripts/train_qwen_law_ft.py \
  --model-name "Qwen/Qwen3-8B" \
  --train-file results/finetune/ft_direct_full_norag.jsonl \
  --output-dir runs/qwen3_law_ft/direct_norag_q8_4bit_v1 \
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
- 実行時の注意:
  - OOM 時はまず `--batch-size 1` に下げるか、`--gradient-accumulation-steps` を増やす。必要なら `max-seq-length` を 768/512 へ縮小。
  - ログ確認: `runs/qwen3_law_ft/direct_norag_q8_4bit_v1` 配下の `trainer_state.json` や `events.out.tfevents.*` を `tail`/TensorBoard で確認。
  - GPU使用率確認: `watch -n1 nvidia-smi` などで VRAM/利用率を監視し、異常があれば batch/seq を調整。
  - 中断時: スクリプトに明示の resume オプションはないため、再実行時は出力先を変えるか、不要なら旧チェックポイントを削除して再実行する。
  - 学習の経過と設定値は `LLM_ファインチューニング/StepD_学習実験ログテンプレート.md` を流用して記録する。
