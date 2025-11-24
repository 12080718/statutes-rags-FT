## A-1 環境要件の棚卸し（Qwen3:8B + RAG想定）

### Pythonパッケージ（主要）
- `langchain`, `langchain-community`（RAGPipelineでOllama LLMを呼び出し）
- `faiss-cpu`（ベクトルインデックス）
- `sentence-transformers`（E5埋め込み、CrossEncoderもここから利用）
- `rank-bm25`（BM25リトリーバ）
- `qdrant-client`（qdrant利用時用、現行はFAISS中心だがpyprojectに含まれる）
- `pydantic` / `python-dotenv` / `tqdm` / `pandas` / `datasets`（設定・IO補助）
- `torch`（CrossEncoderRerankerで必須、GPUがあればCUDA版推奨）
- （dev/optional）`fastapi`, `uvicorn`, `ragas` などはAPI/評価用

### 外部ソフト・モデル
- **Ollama**: LLM実行エンジン。`OLLAMA_HOST` でホスト指定。Qwen3:8B を事前に `ollama pull qwen3:8b` 等で用意。
- **Qwen3:8B モデル**: Ollamaへ登録（ベース & 今後のFTモデルも登録予定）。
- （任意）GPU/CUDA: TorchがGPUを見つければrerankerがGPUを使用。

### 環境変数の想定
- `OLLAMA_HOST`（例: `http://localhost:11434`）: RAGPipelineのLLM接続先。
- `LLM_MODEL`（例: `qwen3:8b`）: `.env` で上書き可能。
- `EMBEDDING_MODEL`（例: `intfloat/multilingual-e5-large`）: VectorRetrieverで使用。
- `VECTOR_STORE_PATH` / `DATA_PATH` など pyproject由来のパス系。
- （OMP関連）`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` などは環境に応じて設定候補（Step3でOMPエラーが出たため対策メモを別途作成）。

### 依存とコード対応箇所メモ
- `app/retrieval/rag_pipeline.py`: `langchain_community.llms.Ollama` 使用 → langchain-community が必要。
- `app/retrieval/vector_retriever.py`: FAISSとsentence-transformers依存。
- `app/retrieval/bm25_retriever.py`: rank-bm25依存。
- `app/retrieval/reranker.py`: torch + sentence-transformers.CrossEncoder 依存（GPUがあれば利用）。
- `scripts/build_finetune_dataset.py`: RAGを使う場合は上記Retriever/RAGPipelineの依存が必要。`--no-rag`なら最小依存で動作。
