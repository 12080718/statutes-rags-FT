## ベースライン評価結果テンプレート（Base: Qwen3:8B）

- 実行日時: YYYY-MM-DD HH:MM (JST)
- 実行者/環境: (例) 自宅PC / GPU: RTXxxxx / RAM: xxGB / OS / Python ver
- 使用モデル:
  - LLM: qwen3:8b (Ollama)
  - 追加: （もしあれば）
- RAG設定:
  - rag_enabled: true/false
  - retriever_type: vector/bm25/hybrid
  - top_k: N
  - reranker: enabled/disabled (model, top_n)
- プロンプト設定:
  - mode: direct / cot
  - few_shot: enabled/disabled
  - ensemble: 1 / 3 / ...
- コマンド（実行に使用したもの）:
  ```
  python scripts/evaluate_multiple_choice.py ...<args>...
  ```
- データセット: lawqa_jp selection (140問) / サブセットの場合は件数
- 結果:
  - accuracy: xx.xx%
  - correct / total: C / T
  - timeout errors: n
  - parse errors (unknown): n
  - other errors: n
- カテゴリ別メモ（任意）:
  - 例: 民法: x/x, 行手法: x/x ...
- 所感/課題:
  - 例: CoTで正答率上がった/下がった、RAG無しとの差分など
- 次のアクション:
  - 例: top_k調整、reranker有効化、FTモデル比較 など
