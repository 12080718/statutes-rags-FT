## StepE まとめ

### 変更・成果物一覧
- スクリプト変更: `scripts/train_qwen_law_ft.py`
  - `--do-train` 追加で明示時のみ学習・保存を実行、デフォルトはセットアップのみ。
  - `--max-steps-override` 追加で step数を上書き可能。
  - 学習本体 (`trainer.train()`, save) は `--do-train` 時にのみ有効化。
- 設計/計画ドキュメント
  - `StepE_タスク1_結果.md`: 本番モード化の方針と推奨手順。
  - `StepE_タスク2_結果.md`: 小規模トライアル学習の計画（10〜20行、max_steps小さめ）。
  - `StepE_タスク3_設計.md`: HFバックエンド追加の設計案（rag_configにbackend/LoRAパス、evaluateで切替）。
  - `StepE_評価コマンド案.md`: HF Base/FT の RAG+direct / RAG+CoT 評価コマンド例。
  - `StepE_評価ログテンプレート.md`: 評価記録用の空テンプレ。

### 学習コマンド案（再掲の要約）
- no-RAG direct 学習例（実行時は `--do-train` 必須）: `train_qwen_law_ft.py ... --train-file ft_direct_full_norag.jsonl --output-dir runs/... --num-epochs 3 --batch-size 2 ... --do-train`
- no-RAG CoT 学習例: 同上で `ft_cot_full_norag.jsonl` を指定。
- トライアル版（E-2）：データを10〜20行にスライス、`--num-epochs 1 --max-steps-override 50 --batch-size 1` などで軽量チェック。

### 評価コマンド案（HFバックエンドを想定）
- Base HF (LoRAなし): RAG+direct / RAG+CoT を `--llm-backend hf --hf-model-name ...` で評価。
- FT HF (LoRAあり): 上記に `--hf-lora-path runs/...` を付与。
- 詳細は `StepE_評価コマンド案.md` を参照。

### 今後、人間が行うべきこと
- HFバックエンド統合の実装（rag_config/rag_pipeline/evaluateへのbackend切替、LoRAロード処理）。
- トライアル学習をGPU環境で実行し、OOM/NaNがないか確認（`--do-train`を付ける）。
- 本番学習の実行パターンを選定（優先: no-RAG direct/CoT）。`trainer.train()`を有効にして実行。
- 学習完了後、HF+LoRAモデルで評価コマンドを実行し、`StepE_評価ログテンプレート.md` に結果を記録。
- 必要ならOllamaバックエンドとの比較も実施し、差分をBベースラインと比較。***
