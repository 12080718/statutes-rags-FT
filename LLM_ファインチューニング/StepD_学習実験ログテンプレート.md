## 学習実験ログテンプレート

- 実験ID:
- 日時:
- GitコミットID:
- 使用モデル: (例) Qwen/Qwen1.5-7B-Chat
- 使用JSONL: (例) results/finetune/ft_direct_full_norag.jsonl
- 主要ハイパラ:
  - num_epochs:
  - batch_size:
  - lr:
  - max_seq_length:
  - lora_r / lora_alpha / lora_dropout:
  - gradient_accumulation_steps / warmup_ratio:
  - fp16/bf16:
  - target_modules:
- 実行コマンド:
  ```
  python scripts/train_qwen_law_ft.py ...<args>...
  ```
- 学習時間:
- 評価コマンド＆結果:
  - 例: `python scripts/evaluate_multiple_choice.py ...` → accuracy:
- ロス推移/ログメモ:
- 所感・次アクション:
