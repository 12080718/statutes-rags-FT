## StepG 変更まとめ（タスクG-1）

- 変更ファイル: `scripts/train_qwen_law_ft.py`
  - 4bit QLoRA対応を追加。
  - 追加引数: `--use-4bit`, `--bnb-4bit-compute-dtype`, `--bnb-4bit-quant-type`。
  - BitsAndBytesConfigで4bitロード（`load_in_4bit=True`, quant_type nf4/fp4, compute dtype selectable）、`prepare_model_for_kbit_training` を適用。
  - 非4bit時は従来の fp16/bf16/dtypeロードを維持。
  - LoRA適用は4bit/非4bit共通。
  - `python -m py_compile` にて構文確認済み。

## 追記（タスクG-2）
- 変更ファイル: `app/llm/hf_llm.py`
  - `HFLLMConfig` に 4bit 用フィールド追加（load_in_4bit, bnb_4bit_compute_dtype, bnb_4bit_quant_type）。
  - 4bitロード: BitsAndBytesConfig で load_in_4bit=True（nf4/fp4、compute dtype選択可）。
  - 非4bit時は従来の fp16/float32 ロードを維持。
  - `python -m py_compile app/llm/hf_llm.py` にて構文確認済み。

## 追記（タスクG-3）
- 変更ファイル: `app/core/rag_config.py`
  - LLMConfigに4bit関連フィールド追加: `load_in_4bit`（env: LOAD_IN_4BIT, default true）、`bnb_4bit_compute_dtype`（env: B2_4BIT_COMPUTE_DTYPE, default bfloat16）、`bnb_4bit_quant_type`（env: B2_4BIT_QUANT_TYPE, default nf4）。
- 変更ファイル: `app/retrieval/rag_pipeline.py`
  - HFバックエンド初期化時に4bit設定を `HFLLMConfig` へ引き渡すように変更。
- 変更ファイル: `scripts/evaluate_multiple_choice.py`
  - CLIに `--hf-load-in-4bit` を追加、指定時に config.llm.load_in_4bit を上書き。
  - 構文チェック `python -m py_compile scripts/evaluate_multiple_choice.py` 済み。

## 追記（タスクG-4）
- インポート/構文スモーク:
  - `python -m py_compile app/llm/hf_llm.py` OK
  - `python -m py_compile scripts/train_qwen_law_ft.py` OK
  - ※ heavyな推論・学習は実行していない。

## 4bit対応に伴うコマンド案の更新
- 学習コマンド案（StepDでの例に4bitオプションを追加）
  - no-RAG direct 学習（4bit有効化例）:
    ```
    python scripts/train_qwen_law_ft.py \
      --model-name Qwen/Qwen1.5-7B-Chat \
      --train-file results/finetune/ft_direct_full_norag.jsonl \
      --output-dir runs/qwen_law_ft/direct_norag_v1_4bit \
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
  - no-RAG CoT 学習でも同様に `--train-file ft_cot_full_norag.jsonl` を指定し、必要に応じてmax_seq_length等を調整。  
  - トライアル時は `--max-steps-override` や小さなデータで試すのは従来通り。

- 評価コマンド案（StepEのHFバックエンド例に4bit設定を追加）
  - HF Base（LoRAなし、4bit強制ON）:
    ```
    python scripts/evaluate_multiple_choice.py \
      --data datasets/lawqa_jp/data/selection.json \
      --output results/evaluations/hf_base_rag_direct_4bit.json \
      --top-k 3 \
      --llm-backend hf \
      --hf-model-name Qwen/Qwen1.5-7B-Chat \
      --hf-load-in-4bit \
      --ensemble 1
    ```
  - HF FT（LoRA適用、4bit強制ON）:
    ```
    python scripts/evaluate_multiple_choice.py \
      --data datasets/lawqa_jp/data/selection.json \
      --output results/evaluations/hf_ft_rag_direct_4bit.json \
      --top-k 3 \
      --llm-backend hf \
      --hf-model-name Qwen/Qwen1.5-7B-Chat \
      --hf-lora-path runs/qwen_law_ft/direct_norag_v1_4bit \
      --hf-load-in-4bit \
      --ensemble 1
    ```
  - CoT評価時は `--use-cot` を付与し、出力ファイル名を `_cot_4bit` にするなど従来と同じ。  
  - Reranker利用時は `--use-reranker --rerank-top-n 3` を追加。  
  - 4bitを無効化したい場合は `--hf-load-in-4bit` を付けない（configのデフォルトに従う）。  
  - GPUメモリが厳しい場合はより小さいモデルやCPU実行、max_new_tokens縮小などの回避策が必要。***

## トライアル実行メモ（no-RAG direct 4bit 学習）
- コマンド: `scripts/train_qwen_law_ft.py --model-name Qwen/Qwen1.5-7B-Chat ... --use-4bit ... --do-train`
- 結果: `prepare_model_for_kbit_training` で CUDA OOM (RTX3080 ~10GB)。モデルロードは完了するが fp32 変換時にメモリ不足。
- 対策候補:
  - より小さいモデル（4B系など）に切り替える。
  - CPU実行またはGPUメモリの大きい環境で試す。
  - 初期化段階でのOOMのため、モデル縮小が現実的。***
