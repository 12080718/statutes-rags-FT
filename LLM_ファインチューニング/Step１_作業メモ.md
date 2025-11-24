## Step1 作業メモ

- 方針確認: `今後の方針.md` を読んで開発ルールとフェーズ方針を整理（既存評価を壊さず、追加モジュールから実装など）。
- コード把握: `app/core/rag_config.py`, `app/retrieval/rag_pipeline.py`, `scripts/evaluate_multiple_choice.py` を読み、各役割と4択RAGフローを整理。
- 設計案: 新規追加予定の `app/core/prompts.py` と `scripts/build_finetune_dataset.py` について、関数/引数/戻り値/呼び出し元の設計を提案。
- 実装追加: `app/core/prompts.py` を新規作成し、4択用プロンプト生成関数 `build_mc_prompt_direct` と `build_mc_prompt_cot`（型ヒント・詳細docstring付き）を実装。既存コードへの差し替えは未着手。

### 変更ファイル
- 追加: `app/core/prompts.py`
- 追加: `LLM_ファインチューニング/Step１.md` (本ファイル)
