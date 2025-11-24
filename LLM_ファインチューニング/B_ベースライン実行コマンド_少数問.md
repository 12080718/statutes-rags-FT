## B-1 少数問ベースライン実行プラン

### CLIオプション要点（scripts/evaluate_multiple_choice.py）
- `--data`: データパス（デフォルト `datasets/lawqa_jp/data/selection.json`）
- `--output`: 結果JSONの出力先
- `--samples`: 評価する件数（先頭からN件にスライス）
- `--no-rag`: RAG無効（LLMのみ）
- `--top-k`: 取得文書数（RAG有効時）。Reranker使用時は自動で大きめに調整。
- `--llm-model`: LLMモデル名（未指定なら config.llm.model_name）
- `--no-few-shot`: Few-shotプロンプト無効
- `--use-reranker`, `--reranker-model`, `--rerank-top-n`: Reranker関連
- `--ensemble`: アンサンブル回数（>1で多数決）
- `--use-cot`: CoTプロンプト有効化

### コマンド例（3〜5問の簡易評価）

#### パターン1: 純LLMベースライン（no-RAG, 3問, Few-shotあり, 直接回答）
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection.json \
  --output results/evaluations/baseline_no_rag_qwen3_3.json \
  --samples 3 \
  --no-rag \
  --llm-model qwen3:8b \
  --top-k 1 \
  --ensemble 1
```

#### パターン2: RAGありベースライン（top_k=3, 5問, Rerankerなし）
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection.json \
  --output results/evaluations/baseline_rag_qwen3_5.json \
  --samples 5 \
  --top-k 3 \
  --llm-model qwen3:8b \
  --ensemble 1
```

#### オプション調整のヒント
- CoT版を試す場合は `--use-cot` を追加。Few-shotを外す場合は `--no-few-shot`。
- Rerankerを使う場合は `--use-reranker --reranker-model <model> --rerank-top-n 3` を追加。
- アンサンブルを試す場合は `--ensemble 3` などに変更。

### 実行前チェック
- RAG有効時: ベクトル/BM25/ハイブリッドのインデックスが `config.vector_store_path` 配下に存在すること（なければ `make index`）。
- Ollama/Qwen3:8B が起動済みで `OLLAMA_HOST` が正しく設定されていること。
- OMP/SHMエラー回避が必要な環境では、必要に応じて `OMP_NUM_THREADS=1` などを事前に設定。
