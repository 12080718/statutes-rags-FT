## ベースライン評価結果（RAGあり・CoTあり・top_k=3・Qwen3:8B）

- 実行日時: （2025/11/24 6:00。ログ上1 run, 約42分）
- 実行者/環境: 横手優一郎（Ollamaローカル / RTX3080）
- 使用モデル:
  - LLM: qwen3:8b (Ollama)
- RAG設定:
  - rag_enabled: true
  - retriever_type: vector
  - top_k: 3
  - reranker: disabled
- プロンプト設定:
  - mode: cot
  - few_shot: enabled（デフォルト）
  - ensemble: 1
- コマンド:
  ```
  python scripts/evaluate_multiple_choice.py \
    --data datasets/lawqa_jp/data/selection.json \
    --output results/evaluations/baseline_rag_qwen3_full_cot.json \
    --top-k 3 \
    --llm-model qwen3:8b \
    --ensemble 1 \
    --use-cot
  ```
- データセット: lawqa_jp selection（140問）
- 結果:
  - accuracy: 60.71%
  - correct / total: 85 / 140
  - timeout errors: 0
  - parse errors (unknown): 2
  - other errors: 0
- カテゴリ別メモ（任意）: 未記入
- 所感/課題:
  - RAG+CoTで direct より精度が向上（52.14% → 60.71%）。パースエラーは2件に減少。
  - LangChainの `Ollama` / `HuggingFaceEmbeddings` は非推奨警告あり（`langchain-ollama` / `langchain-huggingface` への移行検討）。
  - 実行時間は約42分（140問、CoT有効化により direct より長時間）。
- 次のアクション:
  - Reranker有効化/Top_k調整で追加ベースラインを取得。
  - パースエラー削減のためプロンプト微調整を検討。
