## StepH 結果まとめ

### タスクH-1: 小モデル選定
- 選定モデル: `Qwen/Qwen1.5-1.8B-Chat`（約1.8Bパラメータ）
- 理由: 3080 (10GB) 環境で 4bit QLoRA のスモールデバッグを通すため、7Bより小さいモデルが必要なため。***

### タスクH-2: no-RAG direct 小モデル学習スモールラン（提案コマンド）
実行時は `.venv` を有効化し、GPUメモリに応じて batch/max_seq_length を調整してください。まずは以下の軽量設定で試行:
```
python scripts/train_qwen_law_ft.py \
  --model-name Qwen/Qwen1.5-1.8B-Chat \
  --train-file results/finetune/ft_direct_full_norag.jsonl \
  --output-dir runs/debug_small_model/direct_norag_small_4bit \
  --num-epochs 1 \
  --batch-size 2 \
  --learning-rate 2e-4 \
  --max-seq-length 512 \
  --gradient-accumulation-steps 2 \
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
メモ: OOM時は `--batch-size 1` や `--max-seq-length 384/256` へ下げて再試行。4bitで厳しい場合は `--use-4bit` を外して fp16/bf16 で検証する。***

#### 実行結果（実際のラン）
- コマンド: 上記をそのまま実行（Qwen/Qwen1.5-1.8B-Chat, 4bit, max_steps_override=50）
- 実行時間: 約46秒
- ステップ: 50/50 完走
- 最終train_loss: ~1.55
- 出力ディレクトリ: `runs/debug_small_model/direct_norag_small_4bit` にLoRA重み/ログ生成
- 警告: tokenizers fork警告（無視可）、Trainer tokenizer非推奨警告、gradient checkpointingでuse_cache無効化。致命的ではなし。

### タスクH-3: HFバックエンド + 小モデル評価スモーク
- 想定コマンド（no-RAG direct で3問）:
  ```
  python scripts/evaluate_multiple_choice.py \
    --data datasets/lawqa_jp/data/selection.json \
    --output results/evaluations/small_model_debug_direct.json \
    --samples 3 \
    --top-k 1 \
    --no-rag \
    --llm-backend hf \
    --hf-model-name Qwen/Qwen1.5-1.8B-Chat \
    --hf-lora-path runs/debug_small_model/direct_norag_small_4bit \
    --hf-load-in-4bit \
    --ensemble 1
  ```
- 現状: 初回実行で `NameError: llm_config not defined` が発生したため `rag_pipeline.py` を修正（hf初期化用の4bitパラメータを引数で受ける形に変更）。構文チェック済み。
- 次のステップ: 再度上記コマンドで評価を実行し、結果（成功/エラー、精度など）を追記予定。

#### 実行結果（再トライ）
- コマンド: 上記と同一（3問, no-RAG, hf backend, 4bit, LoRA適用）
- 結果: 3問中1問正解（Accuracy 33.33%）、parse unknown 1件、他エラーなし
- 時間: 約4分50秒（3問、1.8B+4bit推論でGPU利用）
- 警告: generateでtemperature/top_p無視の警告、tokenizers fork警告（致命的ではなし）

### StepHまとめ
- 使用小モデル: `Qwen/Qwen1.5-1.8B-Chat`（約1.8B）。3080では7B/8Bの4bitはOOMのため小型に切替。
- 4bit QLoRA学習: 軽量設定（max_steps=50, batch=2, seq=512）でOOMなしに完走（train_loss≈1.55）。7Bは準備段階でOOM。
- HFバックエンド評価: 1.8B+LoRAでno-RAG direct 3問を完走（Accuracy 33.33%、parse unknown 1件）。推論はやや遅い（約5分/3問）。
- 今後A100 20GBでの本番に向けて:
  - nf4+bfloat16の4bit設定はそのまま流用可能。
  - A100ではbatch/seq/max_stepsを拡大してもメモリ余裕がある見込み。
  - HF推論は入力をモデルと同じデバイスに移動する修正済み。
