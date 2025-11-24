## ベースライン評価結果（RAGあり・CoTなし・top_k=3・Qwen3:8B）

- 実行日時: （2025/11/24 5:00, 約30分）
- 実行者/環境: 横手優一郎（Ollamaローカル / RTX3080）
- 使用モデル:
  - LLM: qwen3:8b (Ollama)
- RAG設定:
  - rag_enabled: true
  - retriever_type: vector
  - top_k: 3
  - reranker: disabled
- プロンプト設定:
  - mode: direct
  - few_shot: enabled（デフォルト）
  - ensemble: 1
- コマンド:
  ```
  python scripts/evaluate_multiple_choice.py \
    --data datasets/lawqa_jp/data/selection.json \
    --output results/evaluations/baseline_rag_qwen3_full.json \
    --top-k 3 \
    --llm-model qwen3:8b \
    --ensemble 1
  ```
- データセット: lawqa_jp selection（140問）
- 結果:
  - accuracy: 52.14%
  - correct / total: 73 / 140
  - timeout errors: 0
  - parse errors (unknown): 4
  - other errors: 0
- カテゴリ別メモ（任意）: 未記入
- 所感/課題:
  - LangChainの`Ollama`/`HuggingFaceEmbeddings`に非推奨警告あり（将来`langchain-ollama`/`langchain-huggingface`への移行を検討）。
  - RAGありでも正答率は約52%。パースエラー4件あり（回答がunknownになるケース）。
  - 実行時間は約30分（140問、RAGあり、CoTなし、ensemble=1）。
- 次のアクション:
  - Reranker有効化や top_k 調整のベースライン取得。
  - CoT有効化の比較。
  - パースエラー減少のためのプロンプト微調整を検討。***
