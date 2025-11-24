## A-2 Python依存のインストール方針

### 推奨: 専用 requirements を用意して一括インストール
- ファイル案: `requirements-llm.txt`（まだ作成していません。作成する場合の中身例）  
  ```
  langchain>=0.1.0
  langchain-community>=0.0.10
  faiss-cpu>=1.7.4
  sentence-transformers>=2.2.0
  rank-bm25>=0.2.2
  qdrant-client>=1.7.0
  pydantic>=2.5.0
  python-dotenv>=1.0.0
  tqdm>=4.66.0
  pandas>=2.0.0
  datasets>=2.15.0
  sudachipy>=0.6.0
  sudachidict-core>=20230927
  janome>=0.5.0
  torch  # GPUなら torch==<cuda版> を指定
  ```

- インストールコマンド例（実行はしません）  
  ```
  pip install -r requirements-llm.txt
  ```  
  または pyproject に依存している場合:  
  ```
  pip install .
  ```  
  GPUを使う場合は torch を環境に合わせて指定（例: `pip install torch==2.2.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html`）。

### 代替: 直接インストールする場合のワンライナー例（提案のみ）
```
pip install \
  langchain langchain-community faiss-cpu sentence-transformers rank-bm25 \
  qdrant-client pydantic python-dotenv tqdm pandas datasets \
  sudachipy sudachidict-core janome torch
```

### 留意点
- CrossEncoderRerankerを使う場合は `torch` が必須。GPUがあればCUDA版が高速。
- OllamaはPythonパッケージではなく外部ソフト。`ollama` CLIのインストールと `ollama pull qwen3:8b` が別途必要。
- このドキュメントは提案のみで、インストール作業は実行していません。
