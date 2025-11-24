## StepE-2 小規模トライアル学習計画（実行しない想定）

目的: OOMやNaNが出ないか、パイプラインが通るかを確認するための極小セット試行。

### トライアル前処理案
- データを手元で10〜20行にスライスして別ファイルを作る（例）  
  ```
  head -n 20 results/finetune/ft_direct_full_norag.jsonl > /tmp/ft_direct_trial.jsonl
  head -n 20 results/finetune/ft_cot_full_norag.jsonl > /tmp/ft_cot_trial.jsonl
  ```
  ※ train_qwen_law_ft.py は max-samples 引数を持たないため、事前にファイルを縮小する運用。

### トライアルコマンド案（direct）
```
python scripts/train_qwen_law_ft.py \
  --model-name Qwen/Qwen1.5-7B-Chat \
  --train-file /tmp/ft_direct_trial.jsonl \
  --output-dir runs/trial_direct_norag_v1 \
  --num-epochs 1 \
  --batch-size 1 \
  --learning-rate 5e-5 \
  --max-seq-length 512 \
  --gradient-accumulation-steps 2 \
  --warmup-ratio 0.03 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --max-steps-override 50 \
  --do-train \
  --fp16
```

### トライアルコマンド案（CoT）
```
python scripts/train_qwen_law_ft.py \
  --model-name Qwen/Qwen1.5-7B-Chat \
  --train-file /tmp/ft_cot_trial.jsonl \
  --output-dir runs/trial_cot_norag_v1 \
  --num-epochs 1 \
  --batch-size 1 \
  --learning-rate 5e-5 \
  --max-seq-length 768 \
  --gradient-accumulation-steps 2 \
  --warmup-ratio 0.03 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --max-steps-override 50 \
  --do-train \
  --fp16
```

### 確認したいポイント
- 学習が最後まで通るか（OOM/NaNなし）。
- ロスが減少傾向にあるか（ごく小さなステップでも異常がないか）。
- ログ/出力ディレクトリが正しく生成されるか。

### 予想リソース/時間感（目安）
- GPU: RTX3080クラスで batch_size=1, max_seq_length 512〜768, steps≈50 → 数分〜10分程度。
- CPUのみの場合は時間が大きく延びるのでGPU推奨。

### 実行上の注意
- screen/tmuxでの実行を推奨（接続断対策）。
- OOM時は max_seq_length を下げるか batch_size=1 & grad_accum 増加で対応。
- `--do-train` を忘れると学習が走らないので明示すること。
