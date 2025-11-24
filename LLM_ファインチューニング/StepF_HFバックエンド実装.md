# StepF HFバックエンド実装の指示内容（Phase1）

これから Phase1（HFバックエンドの実装）を進めたいです。  
基本方針・設計は以下のドキュメントに従います。

- `LLM_ファインチューニング/StepE_タスク3_設計.md`
- `LLM_ファインチューニング/StepE_タスク1_結果.md`
- 実装済み:
  - `app/core/rag_config.py`
  - `app/retrieval/rag_pipeline.py`
  - `scripts/evaluate_multiple_choice.py`
  - `scripts/train_qwen_law_ft.py`

---

## 共通ルール

- **新しいブランチ**（例：`feature/hf-backend`）を切って作業してください。  
  `main` を直接変更しないでください。
- 変更は **小さい単位のコミット** に分割してください。
  - 例：
    - rag_config の LLMConfig 拡張 → 1コミット
    - HF LLM モジュール追加 → 1コミット
    - rag_pipeline の分岐追加 → 1コミット
    - evaluate_multiple_choice の CLI オプション追加 → 1コミット
- 既存の Ollama バックエンドは壊さないでください。
  - 何も指定しなければ **今まで通り Ollama を使う** 挙動を維持してください。
- GPU を使う重いコマンド（学習など）は実行しないでください。  
  このステップでは **実装とスモークテスト（ごく軽い実行）まで** にとどめてください。
- 最後に、何を変更したかを  
  `LLM_ファインチューニング/StepF_結果まとめ.md`  
  に Markdown でまとめてください。

---

## タスクF-1：rag_config.py に HF バックエンド用フィールド追加

### 目的

`LLMConfig` に HF バックエンド用の設定項目を追加し、  
`.env` やデフォルト値から設定できるようにします。

### やってほしいこと

1. `app/core/rag_config.py` を開いてください。

2. `LLMConfig`（またはそれに相当する設定クラス）に、以下のフィールドを追加してください。

   - `backend: Literal["ollama", "hf"] = "ollama"`
     - デフォルトは `"ollama"` にしてください（既存挙動を変えないため）。
   - `hf_model_name: Optional[str] = None`
     - 例：`"Qwen/Qwen1.5-7B-Chat"`
   - `lora_path: Optional[str] = None`
     - 例：`"runs/qwen_law_ft/direct_norag_v1"`

3. `.env` や config の読み込みロジック（`load_config()` など）があれば、  
   以下の環境変数から読み取れるようにしてください。

   - `LLM_BACKEND` → `backend`
   - `HF_MODEL_NAME` → `hf_model_name`
   - `LORA_PATH` → `lora_path`

   既存の `LLM_MODEL`（Ollama用）などはそのまま残し、  
   競合しないようにしてください。

4. 変更後、`LLM_ファインチューニング/StepF_結果まとめ.md` に

   - 追加したフィールド名
   - 使う予定の環境変数名
   - デフォルト値

   を箇条書きで記載してください。

---

## タスクF-2：HF LLM モジュールの新規追加（app/llm/hf_llm.py）

### 目的

HF + LoRA モデルをロードし、`RAGPipeline` から呼べる  
**HFバックエンド用 LLM クラス**を追加します。

### やってほしいこと

1. 新規ファイル `app/llm/hf_llm.py` を作成してください。

2. このファイルに、以下のようなクラスを定義してください（クラス名は多少変えても構いません）。

   ```python
   from typing import Optional, Any
   from dataclasses import dataclass

   from transformers import AutoTokenizer, AutoModelForCausalLM
   from peft import PeftModel

   @dataclass
   class HFLLMConfig:
       model_name: str
       lora_path: Optional[str] = None
       device: str = "cuda"  # or "auto"
       max_new_tokens: int = 256
       temperature: float = 0.0

   class HFLoRALLM:
       def __init__(self, config: HFLLMConfig) -> None:
           """
           HF ベースモデルと、必要に応じて LoRA 重みを読み込む。
           """
           ...

       def invoke(self, prompt: str, **kwargs: Any) -> str:
           """
           prompt を入力としてモデルを実行し、日本語テキストを返す。
           kwargs には max_new_tokens, temperature などが渡される可能性がある。
           """
           ...
   ```

3. 実装の要件

   - `__init__` について
     - `AutoTokenizer.from_pretrained(config.model_name, use_fast=True)` で tokenizer をロードする。
     - `AutoModelForCausalLM.from_pretrained(config.model_name, device_map="auto" あるいは config.device)` でモデルをロードする。
     - `config.lora_path` が指定されている場合は、`PeftModel.from_pretrained(model, config.lora_path)` で LoRA を適用する。
     - `self.tokenizer` / `self.model` をインスタンス変数として保持する。

   - `invoke` について
     - 引数：
       - `prompt: str`
       - `**kwargs`: `max_new_tokens`, `temperature` など（指定があれば config を上書き）。
     - 処理：
       1. `prompt` を tokenizer でテンソル化する（`return_tensors="pt"`）。
       2. 必要に応じて `.to(config.device)` でデバイスに載せる。
       3. `model.generate(...)` を呼び出し、`max_new_tokens` / `temperature` などを設定する。
       4. 生成結果からテキストに decode し、**プロンプト部分を除いた生成テキストだけ** を返す。
       5. `eos_token_id`, `pad_token_id` は tokenizer から取得して設定する。
     - 例外が出た場合は、最低限のメッセージを含んだ文字列を返すようにしてもよい。

4. 依存ライブラリ（`transformers`, `peft` など）は  
   既にインストールされている前提で実装してください。  
   インストールコマンドはここでは書かなくて構いません。

5. 実装が終わったら、`LLM_ファインチューニング/StepF_結果まとめ.md` に

   - 追加したクラス名
   - 主なメソッド名（`__init__`, `invoke`）
   - LoRA が有効かどうかをどのように判定しているか

   をまとめてください。

---

## タスクF-3：rag_pipeline.py で backend に応じた LLM を切り替え

### 目的

`RAGPipeline` が、`LLMConfig.backend` に応じて

- `"ollama"` → 既存の Ollama LLM
- `"hf"` → HF+LoRA LLM（タスクF-2で作ったクラス）

を使い分けられるようにします。

### やってほしいこと

1. `app/retrieval/rag_pipeline.py` を開いてください。

2. `RAGPipeline`（または LLM を初期化している箇所）を探し、  
   現在 Ollama を初期化している部分を確認してください。

3. そこに `backend` 分岐を追加してください（例）：

   ```python
   from app.core.rag_config import LLMConfig
   from app.llm.hf_llm import HFLLMConfig, HFLoRALLM

   class RAGPipeline:
       def __init__(self, llm_config: LLMConfig, ...):
           self.backend = llm_config.backend

           if llm_config.backend == "ollama":
               # 既存の Ollama 初期化ロジックをそのまま残す
               self.llm = Ollama(
                   model=llm_config.model,
                   # 既存の引数そのまま
               )
           elif llm_config.backend == "hf":
               if not llm_config.hf_model_name:
                   raise ValueError("hf backend requires hf_model_name in LLMConfig")
               hf_cfg = HFLLMConfig(
                   model_name=llm_config.hf_model_name,
                   lora_path=llm_config.lora_path,
                   # 必要に応じて max_new_tokens などを渡す
               )
               self.llm = HFLoRALLM(hf_cfg)
           else:
               raise ValueError(f"Unknown LLM backend: {llm_config.backend}")
   ```

4. `RAGPipeline` 内で LLM を呼び出している箇所（`self.llm.invoke(...)` や LangChain の `.invoke`）があれば、  
   HF バックエンドでも同じインターフェースで動くように調整してください。

   - 既に `self.llm.invoke(prompt)` 形式で統一されているなら、そのままでOKです。
   - LangChain の LLM クラスと HFLoRALLM のインターフェースに差がある場合は、
     - 必要に応じて HFLoRALLM に「LangChain互換のラッパ」を足すなど、簡単に動く方法を選んでください。

5. ロガーがある場合は、RAGPipeline 初期化時に

   - `backend`
   - `model_name`（Ollama or HF）
   - `lora_path`（HF時のみ）

   をログ出力するようにしてください。

6. 変更内容を、`LLM_ファインチューニング/StepF_結果まとめ.md` に追記してください。

---

## タスクF-4：evaluate_multiple_choice.py に HF 用オプションを追加

### 目的

評価スクリプトから LLM バックエンド（Ollama / HF）を切り替えられるようにします。

### やってほしいこと

1. `scripts/evaluate_multiple_choice.py` を開いてください。

2. CLI オプションに、以下のような引数を追加してください（例）：

   ```python
   parser.add_argument(
       "--llm-backend",
       type=str,
       default=None,  # None の場合は config.llm.backend を使う
       choices=["ollama", "hf"],
       help="LLM backend: ollama or hf",
   )
   parser.add_argument(
       "--hf-model-name",
       type=str,
       default=None,
       help="HF model name when using hf backend",
   )
   parser.add_argument(
       "--hf-lora-path",
       type=str,
       default=None,
       help="LoRA weights path when using hf backend",
   )
   ```

3. `load_config()` や `LLMConfig` を使っている箇所で、  
   CLI オプションが指定された場合はそれを上書きしてください。

   - 例：
     - `if args.llm_backend is not None: config.llm.backend = args.llm_backend`
     - `if args.hf_model_name is not None: config.llm.hf_model_name = args.hf_model_name`
     - `if args.hf_lora_path is not None: config.llm.lora_path = args.hf_lora_path`

4. `RAGPipeline` を初期化する部分では、  
   この更新済み `config.llm` を渡すことで、バックエンドが切り替わるようにしてください。

5. 変更内容と、新しい CLI オプションの説明を  
   `LLM_ファインチューニング/StepF_結果まとめ.md` に追記してください。

---

## タスクF-5：HFバックエンドのスモークテスト

### 目的

HFバックエンドが最低限動くこと（**1問だけでも回答が返ってくること**）を確認します。  
この段階では、精度や速度は気にしなくて構いません。

### やってほしいこと

1. `.env` かコマンドラインから、HF バックエンドを指定してください。  
   例（コマンドライン上書き）：

   ```bash
   python scripts/evaluate_multiple_choice.py      --data datasets/lawqa_jp/data/selection.json      --output results/evaluations/hf_smoke_test.json      --samples 1      --top-k 1      --llm-backend hf      --hf-model-name Qwen/Qwen1.5-7B-Chat      --ensemble 1
   ```

   LoRA はまだ学習していない前提なので、`--hf-lora-path` は指定しなくて構いません。

2. 実行結果として、以下を確認してください。

   - 例外が出ずに終了すること
   - `EVALUATION RESULTS` が表示されること
   - `results/evaluations/hf_smoke_test.json` に 1件分の結果が保存されていること

3. ログに `backend` や `model_name` が表示されていれば、その内容も確認してください。

4. スモークテストの結果（成功/失敗、ログ上の気になる点など）を  
   `LLM_ファインチューニング/StepF_結果まとめ.md` に追記してください。

---

以上のタスクを、この順番で実行してください。  
各タスクの完了時には、`LLM_ファインチューニング/StepF_結果まとめ.md` に

- 変更したファイル
- 追加したクラス・引数・フィールド
- 簡単な動作確認結果

を必ず書き残してください。
