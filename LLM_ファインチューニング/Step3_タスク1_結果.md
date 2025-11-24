## Step3 タスク1 結果メモ

- 追加ファイル: `scripts/build_finetune_dataset.py`
  - lawqa_jp 4択問題からファインチューニング用 JSONL (`{"input","output","meta"}`) を生成するスクリプトを実装。
  - 主な関数:
    - `load_lawqa`: selection.json を読み込み、`{id, question, choices, correct, source_file}` に正規化。
    - `build_context`: RAGPipeline で top_k 件を取得し、`format_context` 済みテキストを返す。
    - `make_instance`: prompts.py を使ってプロンプトを生成し、output/meta を組み立てる（direct は1文字、cot は Reasoning+Answer）。
    - `build_dataset`: サンプル全体を処理し、レコードを yield。
    - `save_jsonl`: レコード列を JSONL 書き出し。
    - `create_retriever`: config に基づき Vector/BM25/Hybrid を構築。
    - `main`: CLI エントリ。lawqa 読み込み→RAG（任意）→レコード生成→保存。
  - CLI 主な引数: `--lawqa-path`, `--output-path`, `--mode direct|cot`, `--top-k`, `--few-shot`, `--no-rag`, `--samples`, `--use-reranker`。

### 次ステップ
- タスク2で `evaluate_multiple_choice.py` を prompts.py 呼び出しに差し替える。
- 小規模サンプルで JSONL 出力と評価スクリプトの動作確認を行う（タスク3）。
