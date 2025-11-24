## StepF 変更まとめ（タスクF-1）

- 変更ファイル: `app/core/rag_config.py`
  - `LLMConfig` に HFバックエンド用フィールドを追加
    - `backend: Literal["ollama","hf"]`（env: `LLM_BACKEND`, default: `"ollama"`）
    - `hf_model_name: Optional[str]`（env: `HF_MODEL_NAME`, default: `None`）
    - `lora_path: Optional[str]`（env: `LORA_PATH`, default: `None`）
  - 既存の Ollama 用フィールド（`provider`, `model_name`, `temperature`, `max_tokens`）は維持。

次ステップ: HFバックエンド実装（HF LLMモジュール追加、rag_pipeline分岐、evaluate CLI拡張など）。***

## 追記（タスクF-2）
- 追加ファイル: `app/llm/hf_llm.py`
  - 追加クラス: `HFLLMConfig`（model_name, lora_path, device, max_new_tokens, temperature）
  - 追加クラス: `HFLoRALLM`
    - `__init__`: tokenizer/modelロード、必要に応じてLoRA適用（lora_path有無で判定）、デバイス配置
    - `invoke(prompt, **kwargs)`: promptをtokenize→generateし、生成部分のみ返す（max_new_tokens/temperature上書き可）

## 追記（タスクF-3）
- 変更ファイル: `app/retrieval/rag_pipeline.py`
  - `RAGPipeline` にバックエンド分岐を追加（ollama/hf）。
  - hf時は `HFLoRALLM` を利用し、LangChain chainは使わず `invoke` を直接呼ぶ。
  - ログに backend/model/lora_path を出力。

## 追記（タスクF-4）
- 変更ファイル: `scripts/evaluate_multiple_choice.py`
  - CLIに `--llm-backend`（ollama/hf）, `--hf-model-name`, `--hf-lora-path` を追加。
  - 指定時に `config.llm` を上書きして RAGPipeline に渡す。
  - 初期化時にバックエンド/HFモデル/LoRAパスを表示。

## スモークテスト（タスクF-5）
- 実行結果: IndentationError が発生し終了（`evaluate_multiple_choice.py` の parser 部分）。  
  対応: インデントを修正し、`python -m py_compile scripts/evaluate_multiple_choice.py` で構文OKを確認済み。  
  再試行時は同コマンドで動作する想定。***
- 追加情報（再実行時）: `ModuleNotFoundError: No module named 'peft'` が発生。HFバックエンドでLoRA有無にかかわらず `peft` が import されるため、実行には `pip install peft` 等で依存を満たす必要あり（ここでは未インストール）。***
- 追加情報（再々実行時）: `.venv` に peft を導入後、HFバックエンドで Qwen/Qwen1.5-7B-Chat をロードし推論を試みたが、RTX3080(約10GB)で GPU メモリ不足による CUDA OOM で生成に失敗。モデルロード自体は成功し、RAG初期化まで進むが推論で落ちる。対策としては小型モデルへの切替、CPU実行、量子化、max_new_tokens縮小等が必要。***
