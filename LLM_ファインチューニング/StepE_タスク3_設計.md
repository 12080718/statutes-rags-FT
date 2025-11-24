## StepE-3 設計: Fine-tuned HF+LoRA モデルを評価に統合する

### 方針
- **案1（採用）: statutes-rags に HFローカルLLMバックエンドを追加**  
  - `rag_config` に LLMバックエンド種別を追加し、`hf` 選択で Transformers+LoRA をロード。  
  - `evaluate_multiple_choice.py` で CLI 切り替えできるようにする（例: `--llm-backend hf`）。  
- 案2（将来案）: LoRA重みをOllamaフォーマットへ変換し`qwen3-law-ft`として登録 → 工数大のため今回は見送り。

### 変更対象ファイル（案1）
- `app/core/rag_config.py`
  - `LLMConfig` に `backend: Literal["ollama","hf"] = "ollama"` を追加。
  - HF 用のモデルID、LoRAパス、デバイス/precisionを指定するフィールド（例: `hf_model_name`, `lora_path`, `hf_dtype`）。環境変数で上書き可。
- `app/retrieval/rag_pipeline.py` または新規モジュール `app/llm/hf_llm.py`
  - `backend=="hf"` の場合に HFモデル+LoRAをロードするクラス/関数を実装。  
  - 推論インターフェースは `.invoke(prompt: str) -> str` に合わせる（Ollama互換）。  
  - トークナイザ/モデルロード: `AutoTokenizer.from_pretrained(hf_model_name)`, `AutoModelForCausalLM.from_pretrained(...)`, `peft.PeftModel.from_pretrained(lora_path)`。
  - 推論時は `generate` を呼び、`max_new_tokens` や `temperature` を `LLMConfig` から参照。
- `scripts/evaluate_multiple_choice.py`
  - CLIに `--llm-backend`（デフォルト "ollama"）と HF関連オプション（model名、loraパス、max_new_tokens など）を追加。
  - パイプライン初期化時に backendを見て LLMを選択。RAG部分はそのまま。

### LoRAパス指定の設計
- `rag_config.LLMConfig` に `lora_path: Optional[str]` を追加し、`.env` で設定可能にする（例: `LLM_LORA_PATH=runs/qwen_law_ft/direct_norag_v1`）。  
- CLIからも `--hf-lora-path` で上書きできるようにする。

### 評価時の挙動
- backend=hf: HF+LoRAモデルで `pipeline.llm.invoke(prompt)` がHF生成をラップしたものになる。  
- backend=ollama: 現行のOllama呼び出しのまま。
- RAG有無は既存オプションで制御（変更不要）。

### 今回は設計のみ、コード変更はこの後のタスクで実施予定。
