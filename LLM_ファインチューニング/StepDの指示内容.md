# StepDの指示内容（Fine-tuning 学習パイプライン構築）

これから StepD（LLMのFine-tuning 学習パイプライン構築）を進めたいです。  
基本方針・設計は以下に書かれている内容に従います。

- `docs/今後の方針.md`
- `LLM_ファインチューニング/Step2の指示内容.md`
- `LLM_ファインチューニング/Step2の結果（設計メモ）`
- `LLM_ファインチューニング/Step3_まとめ.md`
- `LLM_ファインチューニング/A_環境要件まとめ.md`
- `LLM_ファインチューニング/B_ベースライン結果_qwen3_rag_top3_direct.md`
- `LLM_ファインチューニング/B_ベースライン結果_qwen3_rag_top3_cot.md`
- `LLM_ファインチューニング/StepC_まとめ.md`
- 実装済み:
  - `app/core/prompts.py`
  - `scripts/build_finetune_dataset.py`
  - `scripts/evaluate_multiple_choice.py`
  - 4つの JSONL:
    - `results/finetune/ft_direct_full_norag.jsonl`
    - `results/finetune/ft_cot_full_norag.jsonl`
    - `results/finetune/ft_direct_full_rag_top3.jsonl`
    - `results/finetune/ft_cot_full_rag_top3.jsonl`

---

## 共通ルール

- 作業は **既存の作業ブランチ** か、新しいブランチ（例：`feature/llm-ft-train`）で行ってください。  
  `main` を直接変更しないでください。
- このStepDでは **学習スクリプトの設計と実装のみ** を行い、  
  **実際の heavy な学習ジョブの実行は行わないでください**（コマンド案まで）。
- パッケージインストールやGPU設定など、環境に影響する操作は「提案」までにしてください。
- 各タスクの最後に、  
  `LLM_ファインチューニング/StepD_タスクX_結果.md`  
  のような Markdown に「何を決めたか／何を実装したか」をまとめてください。

---

## タスクD-1：Fine-tuning 戦略＆実験パターンの整理

### 目的

どの JSONL を使って、どんなパターンで Fine-tuning を行うかを明確にします。  
（例：direct専用モデル・CoT専用モデル・混合モデル etc.）

### やってほしいこと

1. `StepC_まとめ.md` に記載の4つの JSONL を前提に、まずは以下のような実験パターン案を整理してください。

   - パターンA：no-RAG + direct のみで学習（1文字回答専用モデル）
   - パターンB：no-RAG + CoT のみで学習（Reasoning+Answer モデル）
   - パターンC：no-RAG direct + CoT を混合して学習（multi-task）
   - （余裕があれば）パターンD：RAGあり版 JSONL を使った RAG-aware FT

2. 各パターンについて、

   - 使用する JSONL ファイル名
   - 目的（例：4択精度向上、CoT品質向上、RAG-awareな挙動確認 etc.）
   - 想定する評価方法（`evaluate_multiple_choice.py` での direct / CoT 評価）

   を Markdown の表形式でまとめてください。

3. その表を  
   `LLM_ファインチューニング/StepD_実験計画.md`  
   として保存してください。

> ※ このタスクではまだ「どのパターンで実際に学習するか」を確定しなくても構いません。  
>  候補パターンを出し、優先度のコメントを付けるところまででOKです。

---

## タスクD-2：学習スクリプト train_qwen_law_ft.py のインターフェース設計

### 目的

Qwen 系モデルを LoRA/QLoRA で Fine-tuning するためのスクリプトの  
**引数・入出力・内部構成**を先に決めます。

### やってほしいこと

1. 新規スクリプト `scripts/train_qwen_law_ft.py` を追加する前提で、  
   以下の観点からインターフェース設計を行い、Markdownにまとめてください。

   - 必須引数
     - `--model-name`（ベースとなる HF モデルID、例：`Qwen/Qwen1.5-7B-Chat` ※あくまで例）
     - `--train-file`（JSONLパス：StepC で作成したファイル）
     - `--output-dir`（学習済みLoRA or FTモデルの出力先）
   - 主なオプション引数
     - 学習設定：`--num-epochs`, `--learning-rate`, `--batch-size`, `--max-seq-length` など
     - LoRA設定：`--lora-r`, `--lora-alpha`, `--lora-dropout`, `--target-modules` など
     - 精度・リソース関連：`--gradient-accumulation-steps`, `--warmup-ratio`, `--fp16` or `--bf16`
     - モード切替：
       - `--train-mode direct|cot|auto`（train-file側のmeta.modeを使うかどうか）
   - 想定する入出力形式
     - 入力：`{"input": str, "output": str, "meta": {...}}` のJSONL
     - モデルへの最終的な文字列：
       - シンプルに `input + "\n" + output` を一連のテキストとして学習するか
       - あるいは Qwen のチャット形式（例：`<|im_start|>user ... <|im_start|>assistant ...`）に変換するか

2. 上記を  
   `LLM_ファインチューニング/StepD_タスク2_設計.md`  
   として保存してください。

> ※ このタスクではまだ Python コードは書かないでください。  
>  引数一覧・データフロー・内部ステップを言語化するところまででOKです。

---

## タスクD-3：train_qwen_law_ft.py の実装（コードのみ）

### 目的

タスクD-2で決めたインターフェースに従って、  
実際の学習スクリプト `scripts/train_qwen_law_ft.py` を実装します。

### 実装方針

- ライブラリとしては、以下の組み合わせを想定してください（提案）：
  - `transformers`（`AutoModelForCausalLM`, `AutoTokenizer`）
  - `datasets`（`load_dataset`）
  - `peft`（`LoraConfig`, `get_peft_model`）
  - 必要なら `trl`（`SFTTrainer`）を使っても構いません

- 学習の基本方針：
  - JSONL の各レコードについて、`input` と `output` を結合したテキストを作る
    - 例：`"<USER>\n" + input + "\n<ASSISTANT>\n" + output`
  - 可能であれば「input 部分のトークンには損失をかけず、output 部分のみ損失対象」にする  
    （難しければ、最初のバージョンはテキスト全体に損失をかけても構いません。その場合はコメントに明記してください）
  - LoRA/QLoRA でベースモデルにアダプタを追加して学習

### やってほしいこと

1. `scripts/train_qwen_law_ft.py` を新規作成し、以下の要素を含む Python コードを実装してください。

   - `argparse` による CLI 引数処理（タスクD-2で決めたもの）
   - JSONL の読み込み（`datasets.load_dataset("json", ...)` を利用）
   - トークナイズ & データセット変換
   - モデル・トークナイザのロード
   - LoRA/QLoRA の設定と適用
   - `Trainer` or `SFTTrainer` による学習ループ実装
   - `output_dir` への保存（LoRAのアダプタ重み、config など）

2. このタスクでは、**実際に学習を走らせるコードを呼び出さないでください。**  
   スクリプト本体の実装のみ行い、実行は後のステップ（人間）に任せてください。

3. 実装が終わったら、主な関数・クラス・処理フローを  
   `LLM_ファインチューニング/StepD_タスク3_結果.md`  
   に要約してください。

---

## タスクD-4：学習実行コマンド案と実験ログテンプレートの作成

### 目的

あとで GPU 上で学習を回すときに、そのまま使える **学習コマンド例** と、  
各実験のログを記録する **テンプレート** を用意しておきます。

### やってほしいこと

1. タスクD-1で整理した実験パターンのうち、  
   「まず最初に試す候補」を 1〜2 個決めてください（例：パターンAとB）。

2. 各候補について、`train_qwen_law_ft.py` を使った学習コマンド例を作成してください。

   - 例（あくまで例です）：
     ```bash
     python scripts/train_qwen_law_ft.py \
       --model-name Qwen/Qwen1.5-7B-Chat \
       --train-file results/finetune/ft_direct_full_norag.jsonl \
       --output-dir runs/qwen_law_ft/direct_norag_v1 \
       --num-epochs 3 \
       --batch-size 2 \
       --learning-rate 2e-4 \
       --max-seq-length 1024 \
       --lora-r 32 \
       --lora-alpha 64 \
       --lora-dropout 0.05
     ```
   - 実際のハイパーパラメータは、RTX3080 クラスの GPU を想定して無理のない値を提案してください。

3. `LLM_ファインチューニング/StepD_学習コマンド案.md` に、  
   上記コマンド例と簡単な注意点（メモリ使用量の見積もりなど）をまとめてください。

4. 別ファイルとして、  
   `LLM_ファインチューニング/StepD_学習実験ログテンプレート.md`  
   を作成し、各学習実験のログを書き込むためのテンプレートを作ってください。

   - 含めたい項目例：
     - 実験ID
     - 日時
     - GitのコミットID
     - 使用モデル名
     - 使用した JSONL ファイル
     - 主要ハイパーパラメータ
     - 学習時間
     - 学習後に実行した評価コマンド＆結果（accuracyなど）
     - 所感・次のアクション

5. このタスクでも、**実際の学習コマンドは実行しないでください**。  
   あくまで「準備」と「テンプレート作成」までです。

---

以上のタスクを、この順番で実行してください。  
各タスクの完了時には、対応する `StepD_タスクX_結果.md` などに  
「何を決めたか／何を実装したか」を必ず記録してください。
