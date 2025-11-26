# StepI Qwen3:8B 4bit QLoRA 基本設定案（A100 20GB 専用）

## 前提・目的
- heart01（A100 20GB/MIG）で Hugging Face バックエンド + 4bit QLoRA を使い、Qwen3:8B 系モデルを no-RAG direct で本番学習するための設定テンプレート。
- モデル名は `<HF_QWEN3_8B_MODEL_NAME>` のプレースホルダで扱い、実際のモデルは heart01 上でユーザが決定する。
- 学習スクリプトは `scripts/train_qwen_law_ft.py` を想定。RTX3080 ではこの設定を使わない（メモリ不足のため A100 専用）。

## 基本設定（案）
- `model-name`: `<HF_QWEN3_8B_MODEL_NAME>`
- `train-file`: `results/finetune/ft_direct_full_norag.jsonl`
- `output-dir`: `runs/qwen3_law_ft/direct_norag_q8_4bit_v1`
- `num-epochs`: 3
- `batch-size`: 2（A100 20GB 前提。必要に応じ heart01 で調整）
- `max-seq-length`: 1024
- `gradient-accumulation-steps`: 4
- `learning-rate`: 2e-4
- `warmup-ratio`: 0.03
- `lora-r`: 16
- `lora-alpha`: 32
- `lora-dropout`: 0.05
- `use-4bit`: true
- `bnb-4bit-compute-dtype`: bfloat16
- `bnb-4bit-quant-type`: nf4

## 備考
- heart01 上での GPU メモリ状況に応じて `batch-size` や `max-seq-length` は調整可。OOM 時は batch を 1 へ下げる、もしくは `gradient-accumulation-steps` を増やす。
- トライアル（短時間動作確認）では `--max-steps-override 50` などを併用し、本番学習時のみ `num-epochs=3` とする想定。
- CoT や RAG バリエーションを行う場合は、対応する JSONL に `train-file` を差し替え、出力ディレクトリ名も用途に合わせて枝番を付ける。
