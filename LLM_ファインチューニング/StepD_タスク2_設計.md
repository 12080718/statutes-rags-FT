## StepD-2 `train_qwen_law_ft.py` インターフェース設計

### 必須引数
- `--model-name` : ベースHFモデルID（例: `Qwen/Qwen1.5-7B-Chat`）
- `--train-file` : 学習用JSONL（`{"input","output","meta"}`）
- `--output-dir` : 学習済みLoRA/FTの保存先

### 主なオプション
- 学習設定: `--num-epochs`, `--learning-rate`, `--batch-size`, `--max-seq-length`, `--gradient-accumulation-steps`, `--warmup-ratio`, `--weight-decay`
- 精度/リソース: `--fp16`, `--bf16`
- LoRA設定: `--lora-r`, `--lora-alpha`, `--lora-dropout`, `--target-modules` (カンマ区切り), `--lora-bias` (none/lora/all)
- モード切替: `--train-mode direct|cot|auto`（auto時はmeta.modeを参照）
- ロス設定: `--loss-on-output-only`（input部分をマスクできる場合のみ）
- その他: `--logging-steps`, `--save-steps`, `--eval-steps`（評価は任意）

### 入出力形式
- 入力JSONL: 各行 `{ "input": str, "output": str, "meta": {...} }`
- モデルに渡す最終テキスト: デフォルトはシンプル連結  
  `"<USER>\n{input}\n<ASSISTANT>\n{output}"`  
  （チャット形式に寄せるが、システムプロンプト等は付けない簡易形）
- ロス計算: `--loss-on-output-only` 有効時は `<ASSISTANT>` 以降のトークンのみ損失対象（tokenizerのoffset_mappingが使えるfast tokenizer前提）。無効時は全トークンに損失。

### 内部ステップ（想定）
1. argparseで引数を取得
2. `datasets.load_dataset("json", data_files=train_file)` で読み込み
3. サンプルごとに上記フォーマットでテキスト生成
4. トークナイズ（`max_length`，`truncation=True`）
   - `loss-on-output-only`時はoffset_mappingから `<ASSISTANT>` 位置より前を `-100` でマスク
5. モデル/トークナイザを `AutoModelForCausalLM`, `AutoTokenizer` でロード（`padding_side="right"`, `trust_remote_code` は任意）
6. LoRA設定を `peft.LoraConfig` → `get_peft_model` で適用
7. `Trainer` または `SFTTrainer` で学習  
   - 今回は標準 `Trainer` を想定（シンプルなtext-to-text）。evalは省略可。
8. `output_dir` にアダプタ重み・トークナイザ・configを保存

### 注意
- heavyジョブは実行しない。スクリプト実装のみ。
- `--train-mode auto` 時は meta.mode が direct/cot 混在のデータにも対応可能。
