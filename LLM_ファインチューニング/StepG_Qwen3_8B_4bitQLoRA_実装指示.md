# StepG Qwen3:8B 4bit QLoRA 実装の指示内容

これから Phase1 の延長として、**Qwen3:8B を 4bit QLoRA 前提で扱うための実装**を行いたいです。  
ここではコード実装のみを行い、**実際の学習・長時間推論は行わない**でください。

基本方針・設計は以下のドキュメントとコードに従います。

- `LLM_ファインチューニング/StepE_タスク3_設計.md`
- `LLM_ファインチューニング/StepE_タスク1_結果.md`
- `LLM_ファインチューニング/StepF_結果まとめ.md`
- 実装済み:
  - `app/core/rag_config.py`
  - `app/retrieval/rag_pipeline.py`
  - `app/llm/hf_llm.py`
  - `scripts/evaluate_multiple_choice.py`
  - `scripts/train_qwen_law_ft.py`

ここでのゴールは：

- 学習用スクリプト `train_qwen_law_ft.py`
- 推論用 HF バックエンド `HFLoRALLM`（`app/llm/hf_llm.py`）

の両方が、

> 「Qwen3:8B を 4bit QLoRA でロード・学習・推論できる状態」

になるようにすることです。

---

## 共通ルール

- **新しいブランチ**（例：`feature/qwen3-8b-4bit`）を切って作業してください。  
  `main` を直接変更しないでください。
- 変更は **小さい単位のコミット** に分割してください。
  - 例：
    - train スクリプトの 4bit 対応 → 1コミット
    - HFLoRALLM の 4bit 対応 → 1コミット
- 既存の fp16 / full precision のコードパスは壊さないようにし、  
  4bit は「オプションで有効化できる」形にしてください。
- GPU を使う重いコマンド（長時間の学習・140問評価など）は実行しないでください。  
  このステップでは **実装と軽い import / py_compile レベルの確認**までにとどめてください。
- 最後に、何を変更したかを  
  `LLM_ファインチューニング/StepG_結果まとめ.md`  
  に Markdown でまとめてください。

---

## タスクG-1：train_qwen_law_ft.py を 4bit QLoRA 対応にする

### 目的

学習用スクリプト `scripts/train_qwen_law_ft.py` を、  
**Qwen3:8B を 4bit QLoRA でロード・学習できるようにする**。

### やってほしいこと

1. `scripts/train_qwen_law_ft.py` を開き、既存のモデルロード部分を確認してください。

   現状はおそらく：

   - `AutoModelForCausalLM.from_pretrained(...)` で fp16 or full precision をロード
   - その後 LoRA を適用（`get_peft_model` 等）

   という流れになっているはずです。

2. 4bit 関連の import を追加してください。

   ```python
   import torch
   from transformers import BitsAndBytesConfig
   from peft import prepare_model_for_kbit_training
   ```

3. argparse に、4bit 関連のオプションを追加してください。

   例：

   ```python
   parser.add_argument(
       "--use-4bit",
       action="store_true",
       help="Enable 4-bit QLoRA training via bitsandbytes.",
   )
   parser.add_argument(
       "--bnb-4bit-compute-dtype",
       type=str,
       default="bfloat16",
       choices=["float16", "bfloat16", "float32"],
       help="Compute dtype for 4-bit layers.",
   )
   parser.add_argument(
       "--bnb-4bit-quant-type",
       type=str,
       default="nf4",
       choices=["nf4", "fp4"],
       help="Quantization type for 4-bit weights.",
   )
   ```

4. `main()` 内で、`args.use_4bit` が指定された場合の  
   `BitsAndBytesConfig` を用意してください。

   例：

   ```python
   compute_dtype_map = {
       "float16": torch.float16,
       "bfloat16": torch.bfloat16,
       "float32": torch.float32,
   }
   bnb_config = None
   if args.use_4bit:
       bnb_config = BitsAndBytesConfig(
           load_in_4bit=True,
           bnb_4bit_use_double_quant=True,
           bnb_4bit_quant_type=args.bnb_4bit_quant_type,
           bnb_4bit_compute_dtype=compute_dtype_map[args.bnb_4bit_compute_dtype],
       )
   ```

5. モデルロード部分を、`use-4bit` の有無で分岐させてください。

   - `use-4bit == False` の場合：
     - 既存の `AutoModelForCausalLM.from_pretrained(...)` のコードパスを維持
   - `use-4bit == True` の場合：
     - 以下のように 4bit 用のロードを行う

   例：

   ```python
   if args.use_4bit:
       model = AutoModelForCausalLM.from_pretrained(
           args.model_name,
           quantization_config=bnb_config,
           device_map="auto",
       )
       model = prepare_model_for_kbit_training(model)
   else:
       model = AutoModelForCausalLM.from_pretrained(
           args.model_name,
           torch_dtype=torch.float16,
           device_map="auto",
       )
   ```

6. LoRA の適用処理（`get_peft_model` など）は、  
   **4bit / 非4bit の両方で同じように動くように**維持してください。

7. 最後に、`python -m py_compile scripts/train_qwen_law_ft.py` で  
   構文エラーがないことを確認してください（実行は不要）。

8. 変更内容を `LLM_ファインチューニング/StepG_結果まとめ.md` に箇条書きで記載してください。

---

## タスクG-2：HFLoRALLM（推論用）を 4bit 対応にする

### 目的

`app/llm/hf_llm.py` 内の `HFLoRALLM` が、  
推論時にも **4bit で Qwen3:8B をロードできるようにする**。

### やってほしいこと

1. `app/llm/hf_llm.py` を開き、`HFLLMConfig` / `HFLoRALLM` を確認してください。

2. `HFLLMConfig` に 4bit 用のフィールドを追加してください。

   例：

   ```python
   @dataclass
   class HFLLMConfig:
       model_name: str
       lora_path: Optional[str] = None
       device: str = "cuda"  # or "auto"
       max_new_tokens: int = 256
       temperature: float = 0.0
       load_in_4bit: bool = True
       bnb_4bit_compute_dtype: str = "bfloat16"
       bnb_4bit_quant_type: str = "nf4"
   ```

3. 4bit 関連の import を追加してください（まだであれば）。

   ```python
   import torch
   from transformers import BitsAndBytesConfig
   ```

4. `HFLoRALLM.__init__` で、`config.load_in_4bit` を見て  
   モデルロードの方法を分岐させてください。

   - `load_in_4bit == True` の場合：

     ```python
     compute_dtype_map = {
         "float16": torch.float16,
         "bfloat16": torch.bfloat16,
         "float32": torch.float32,
     }
     bnb_config = BitsAndBytesConfig(
         load_in_4bit=True,
         bnb_4bit_use_double_quant=True,
         bnb_4bit_quant_type=config.bnb_4bit_quant_type,
         bnb_4bit_compute_dtype=compute_dtype_map[config.bnb_4bit_compute_dtype],
     )
     self.model = AutoModelForCausalLM.from_pretrained(
         config.model_name,
         quantization_config=bnb_config,
         device_map="auto" if config.device == "auto" else {0: config.device},
     )
     ```

   - `load_in_4bit == False` の場合：
     - 既存の fp16 / full precision ロードロジックをそのまま使用。

5. LoRA の適用（`config.lora_path` が指定されている場合）は、  
   4bit / 非4bit どちらでも同様に適用されるよう維持してください。

6. `invoke` メソッドでは、既存のロジック（tokenize → generate → decode）を維持しつつ、  
   4bit モデルでも問題なく動くようにしてください（特に変更不要でも構いません）。

7. 最後に、`python -m py_compile app/llm/hf_llm.py` で  
   構文エラーがないことを確認してください。

8. 変更内容を `LLM_ファインチューニング/StepG_結果まとめ.md` に追記してください。

---

## タスクG-3：rag_config.py / evaluate_multiple_choice.py との整合性確認

### 目的

- 4bit 関連の設定が `rag_config.py` → `HFLLMConfig` → `HFLoRALLM`  
  に自然に流れるようにする。
- 評価スクリプトからも 4bit モードを選べるようにする。

### やってほしいこと

1. `app/core/rag_config.py` の `LLMConfig` に、  
   必要であれば 4bit 関連フィールドを追加してください。

   例：

   ```python
   class LLMConfig(BaseModel):
       ...
       load_in_4bit: bool = True
       bnb_4bit_compute_dtype: str = "bfloat16"
       bnb_4bit_quant_type: str = "nf4"
   ```

   可能であれば `.env` からも指定できるようにしつつ、  
   デフォルトは上記のような**QLoRA標準設定**で構いません。

2. `rag_pipeline.py` で HF バックエンドを初期化している部分を確認し、  
   `HFLLMConfig` へ 4bit 設定を引き渡すように修正してください。

   例：

   ```python
   hf_cfg = HFLLMConfig(
       model_name=llm_config.hf_model_name,
       lora_path=llm_config.lora_path,
       device="auto",
       max_new_tokens=llm_config.max_new_tokens,
       temperature=llm_config.temperature,
       load_in_4bit=llm_config.load_in_4bit,
       bnb_4bit_compute_dtype=llm_config.bnb_4bit_compute_dtype,
       bnb_4bit_quant_type=llm_config.bnb_4bit_quant_type,
   )
   ```

3. `scripts/evaluate_multiple_choice.py` に、  
   必要ならば 4bit の ON/OFF を制御する CLI オプションを追加してください。

   例：

   ```python
   parser.add_argument(
       "--hf-load-in-4bit",
       action="store_true",
       help="Use 4-bit quantization for HF backend.",
   )
   ```

   その上で、`args.hf_load_in_4bit` が指定された場合は  
   `config.llm.load_in_4bit = True` のように設定を上書きしてください。  
   （デフォルトで True でも構いませんが、CLIから強制ONできるようにしておくと便利です。）

4. 変更後、以下の 2つのファイルを py_compile で確認してください。

   ```bash
   python -m py_compile app/core/rag_config.py
   python -m py_compile scripts/evaluate_multiple_choice.py
   ```

5. 整合性確認の結果と、追加したフィールド・オプション名を  
   `LLM_ファインチューニング/StepG_結果まとめ.md` に追記してください。

---

## タスクG-4：軽いインポートテスト（任意のスモーク）

### 目的

4bit 対応後のコードが **少なくとも import レベルでは問題ない**ことを確認します。

### やってほしいこと

1. 以下のようなワンライナーを Python から実行し、  
   import エラーが出ないことを確認してください（実行は最小限）。

   ```bash
   python -c "from app.llm.hf_llm import HFLLMConfig, HFLoRALLM; print('hf_llm import OK')"
   python -c "from scripts import train_qwen_law_ft; print('train_qwen_law_ft import OK')"
   ```

2. CUDA / VRAM を大量に使うような処理は走らせないでください。  
   あくまで「import が通るか」「クラスの初期化まで行けるか」程度に留めてください。

3. 結果を `LLM_ファインチューニング/StepG_結果まとめ.md` に追記してください。

---

以上のタスクを、この順番で実行してください。  
各タスクの完了時には、`LLM_ファインチューニング/StepG_結果まとめ.md` に

- 変更したファイル
- 追加したクラス・引数・フィールド
- 簡単な動作確認結果

を必ず書き残してください。
