## 評価ログテンプレート（Base vs FT）

- 実験ID:
- 日時:
- GitコミットID:
- 使用モデル:
  - backend: ollama / hf
  - baseモデル: (例) qwen3:8b or Qwen/Qwen1.5-7B-Chat
  - LoRA: path or none
- retriever設定:
  - rag_enabled: true/false
  - retriever_type: vector/bm25/hybrid
  - top_k:
  - reranker: enabled/disabled (model, top_n)
- プロンプト設定:
  - mode: direct / cot
  - few_shot: enabled/disabled
  - ensemble: 1 / 3 / ...
- 実行コマンド:
  ```
  python scripts/evaluate_multiple_choice.py ...<args>...
  ```
- 結果:
  - accuracy:
  - correct / total:
  - timeout errors:
  - parse errors:
  - other errors:
- 比較対象:
  - baseとの差分、改善/悪化ポイント
- 所感・次の改善:
