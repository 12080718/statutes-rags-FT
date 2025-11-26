# StepI A100 Qwen3:8B 4bit 評価指示（heart01用）

## 3-1. 評価の目的
- no-RAG direct で、HF Qwen3:8B ベースライン（`Qwen/Qwen3-8B`）と 4bit QLoRA 学習済みモデルの140問精度を比較し、改善度とエラー傾向を確認する。
- heart01（A100 20GB/MIG）で実行し、HF バックエンドのみ使用（Ollama/ curl 不使用）。

## 3-2. ベースライン評価（LoRAなし）
- 目的: `<HF_QWEN3_8B_MODEL_NAME>` の素モデル性能を把握。
- コマンド例:
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection.json \
  --output results/evaluations/qwen3_hf_norag_direct_140.json \
  --samples 140 \
  --top-k 1 \
  --no-rag \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-load-in-4bit \
  --ensemble 1
```
- 記録: `LLM_ファインチューニング/B_ベースライン結果テンプレート.md` などに精度・出力傾向を記載。必要に応じ `--no-few-shot` でプロンプトを簡素化し、出力崩れを抑制。

## 3-3. FT後モデル評価（LoRA適用）
- 目的: 学習済み LoRA（`runs/qwen3_law_ft/direct_norag_q8_4bit_v1`）適用時の改善度を測る。
- コマンド例:
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection.json \
  --output results/evaluations/qwen3_hf_ft_norag_direct_140.json \
  --samples 140 \
  --top-k 1 \
  --no-rag \
  --llm-backend hf \
  --hf-model-name "Qwen/Qwen3-8B" \
  --hf-lora-path runs/qwen3_law_ft/direct_norag_q8_4bit_v1 \
  --hf-load-in-4bit \
  --ensemble 1
```
- 2-3. トライアル学習（50ステップ）で作ったLoRAを試す場合は、上記コマンドの `--hf-lora-path` を `runs/qwen3_law_ft/direct_norag_q8_4bit_trial` に差し替える（評価サンプル数は本番同様140のままでよい）。
- 記録: `LLM_ファインチューニング/StepI_評価結果_direct.md` などに正答率・出力崩れ・推論時間をメモ。ベースラインとの差分を併記。

## 3-4. 比較メモのテンプレ（追記用）
- 精度比較: ベースライン vs FT（%表示、差分）。
- 改善/悪化した設問の傾向: 条文/定義/数値問題など。
- 出力品質: parse unknown / 形式崩れの頻度、`--no-few-shot` 有無の影響。
- 今後の拡張メモ: RAGやCoT追加時に注目すべき失点パターン。
