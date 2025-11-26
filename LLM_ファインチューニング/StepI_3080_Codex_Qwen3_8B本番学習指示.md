# StepI（Qwen3:8B 本番学習）用・3080側 Codex 指示書

このドキュメントは、**ローカル RTX3080 + Codex** 環境で動いている Codex に対して、  
次フェーズ（StepI：Qwen3:8B 本番学習）に向けた作業内容を指示するものです。

> ※この指示書は「新しい Codex チャット」に渡されることを想定しています。  
> 過去の会話履歴は引き継がれないため、**リポジトリ内の md / py ファイルだけを手がかりに状況を理解する**前提で作業してください。

---

## 0. 前提とゴール

### 前提

- リポジトリ（例）：`statutes-rags-FT`（ローカルパスは VSCode 側の設定に従う）
- これまでに以下が実装済み：
  - `app/core/prompts.py`：4択問題用 direct / CoT プロンプト
  - `scripts/build_finetune_dataset.py`：FT 用 JSONL 生成（no-RAG / direct / CoT）
  - `scripts/evaluate_multiple_choice.py`：prompts.py を利用した評価スクリプト
  - `scripts/train_qwen_law_ft.py`：HF + QLoRA（4bit）での学習スクリプト
  - `app/llm/hf_llm.py`：HF バックエンド（HFLLM / HFLoRALLM、4bit対応）
- heart01（A100 20GB/MIG）側では、以下が**すでに確認済み**：
  - 1.8B 小モデル（`Qwen/Qwen1.5-1.8B-Chat`）+ 4bit QLoRA で
    - `ft_direct_full_norag.jsonl` を使った no-RAG direct 学習
    - HF バックエンド経由での評価
    が一通り動作すること（パイプライン検証完了）。

### ゴール（StepI 全体）

- **heart01（A100 20GB）上で「Qwen3:8B 系 HF モデル」を 4bit QLoRA で本番学習・評価するための指示 md とコマンド群を、3080 + Codex 側で設計・準備する。**
- 3080 上の Codex は **コード編集と「heart01用手順書（md）」の作成**に専念し、  
  実際の学習・評価コマンドは heart01 側でユーザが叩く前提。

---

## 1. 共通ルール（Codex へのお願い）

1. **heart01 上では Codex や Ollama は使わない**  
   - heart01 は「本番計算専用」環境。
   - Codex は「ローカル RTX3080 環境での開発」にのみ使い、heart01 用の md / コマンド例を出力する役割を担う。

2. **heart01 では Hugging Face バックエンドのみ使用**  
   - `--llm-backend hf`  
   - `--hf-model-name` / `--hf-lora-path` / `--hf-load-in-4bit`  
   を用いる。
   - heart01 では Ollama を前提としたコマンドや curl ベースのインストール例は**提示しない**。

3. **curl 禁止 → wget / pip で完結する例のみを提示**  
   - heart01 用のセットアップ手順を書く場合は、
     - `pip install ...`
     - `wget ... && bash ...`
     のような形にする。
   - `curl ... | sh` 形式は使用しない。

4. **この StepI では基本的に「コード改造は最小限」**  
   - すでに small-model でパイプライン検証が終わっているため、
   - Qwen3:8B 本番学習のためにどうしても必要な修正のみ行い、
   - それ以外は **md での計画・コマンド設計**に専念する。

---

## 2. タスクI-0：前提情報の確認と簡易まとめ

### やること

1. リポジトリ内の次のファイルを読み、heart01とローカルの運用方針・現状を把握してください。

   - `LLM_ファインチューニング/heart01_vs_local_運用方針_Codex用.md`
   - `LLM_ファインチューニング/Heart01_setup_report.md`  
     （※実際のファイル名は `Heart01_setup_report.md` など、近い名前を検索して確認してください）
   - `LLM_ファインチューニング/StepH_結果まとめ.md`
   - `LLM_ファインチューニング/StepG_結果まとめ.md`
   - `LLM_ファインチューニング/StepD_実験計画.md`（あれば）

2. 上記を読んだうえで、以下を**3〜7行程度の箇条書き**で整理した md を新規作成してください。

   - 保存先（推奨）：  
     `LLM_ファインチューニング/StepI_前提整理.md`
   - 内容の例：
     - heart01 とローカルの役割分担
     - 既に small-model（1.8B）で確認されたこと
     - Qwen3:8B 本番学習・評価で目指すこと

---

## 3. タスクI-1：HF Qwen3:8B モデル名と設定の設計

### 目的

heart01 上で使用する **HF の Qwen3:8B 系モデル名と学習設定**を設計し、  
今後のコマンドや評価テンプレで一貫して使える形にする。

### やること

1. HF の Qwen3:8B 相当モデル名を **変数的に扱う**設計にしてください。
   - 例：`<HF_QWEN3_8B_MODEL_NAME>`
   - 実際のモデル名はユーザが heart01 上で決定する前提とし、  
     **Codex 側では決め打ちしない**でください。

2. `scripts/train_qwen_law_ft.py` の `--model-name` にこのプレースホルダを渡す前提で、
   以下のような **基本設定案**をまとめた md を作成してください。

   - 保存先（推奨）：  
     `LLM_ファインチューニング/StepI_Qwen3_8B_設定案.md`
   - 設定例（値は案として記述）：
     - `model-name`: `<HF_QWEN3_8B_MODEL_NAME>`
     - `train-file`: `results/finetune/ft_direct_full_norag.jsonl`
     - `output-dir`: `runs/qwen3_law_ft/direct_norag_q8_4bit_v1`
     - `num-epochs`: 3
     - `batch-size`: 2（A100 20GB想定、必要なら heart01 で調整）
     - `max-seq-length`: 1024
     - `gradient-accumulation-steps`: 4
     - `learning-rate`: `2e-4`
     - `warmup-ratio`: `0.03`
     - `lora-r`: 16
     - `lora-alpha`: 32
     - `lora-dropout`: 0.05
     - `use-4bit`: true
     - `bnb-4bit-compute-dtype`: `bfloat16`
     - `bnb-4bit-quant-type`: `nf4`

3. md には「RTX3080 上ではこの設定を使わないこと（A100 専用設定であること）」も明記してください。

---

## 4. タスクI-2：heart01 用「本番学習指示」md の作成

### 目的

heart01 上でユーザが **Qwen3:8B 4bit QLoRA（no-RAG direct）本番学習**を実行するための  
具体的な手順書（StepI 本番学習指示 md）を作る。

### やること

1. 次のような md を新規作成してください。

   - 推奨ファイル名：  
     `LLM_ファインチューニング/StepI_A100_Qwen3_8B_本番学習指示.md`

2. md の構成（例）：

   #### 2-1. 目的

   - Qwen3:8B 系 HF モデルを A100 20GB 上で 4bit QLoRA 学習し、  
     lawqa_jp 140問（no-RAG direct）に対する性能を向上させること。

   #### 2-2. 前提環境の確認

   - `nvidia-smi` の例（MIG 20GB スライス）
   - `python -m venv .venv` / `pip install -r requirements-llm.txt` が完了していること
   - `datasets/lawqa_jp/data/selection.json` と  
     `results/finetune/ft_direct_full_norag.jsonl` が存在すること

   #### 2-3. トライアル学習コマンド（50ステップ程度）

   - `<HF_QWEN3_8B_MODEL_NAME>` を使った短い学習コマンド例を示す：
     - `--max-steps-override 50`（あれば）
     - `num-epochs 1`
     - `batch-size 1〜2`
     - `max-seq-length 512`
   - 実行後に確認すべきログ（loss, 所要時間）を箇条書きで記載。
   - 結果は `LLM_ファインチューニング/StepI_タスク2_結果.md` 等にメモするよう指示。

   #### 2-4. 本番学習コマンド（3エポック）

   - `StepI_Qwen3_8B_設定案.md` で決めた設定を元にしたコマンド例を記載。
   - 実行時の注意：
     - 途中で学習が落ちた場合の再開方法（checkpoint から）
     - ログ・GPU使用率の確認方法
   - 学習ログは `StepD_学習実験ログテンプレート.md` など既存テンプレートを再利用する想定で明記。

3. **重要**：この md では「heart01 上でユーザが実行するコマンド例のみ」を書き、  
   Codex 自身がコマンドを実行する前提では書かないでください。

---

## 5. タスクI-3：heart01 用「評価指示」md の作成

### 目的

heart01 上で、**Qwen3:8B ベースライン vs Qwen3:8B-FT（direct no-RAG）**の  
140問評価を実行・比較するための手順書を作る。

### やること

1. 次のような md を新規作成してください。

   - 推奨ファイル名：  
     `LLM_ファインチューニング/StepI_A100_Qwen3_8B_評価指示.md`

2. md の構成例：

   #### 3-1. 評価の目的

   - no-RAG direct における
     - ベースライン Qwen3:8B（HF）
     - QLoRA-FT後 Qwen3:8B-FT
   - の正答率・エラー傾向を比較する。

   #### 3-2. ベースライン評価コマンド

   - `scripts/evaluate_multiple_choice.py` を HF バックエンドモードで実行：
     - `--llm-backend hf`
     - `--hf-model-name <HF_QWEN3_8B_MODEL_NAME>`
     - `--samples 140`
     - `--no-rag`
   - 出力例：`results/evaluations/qwen3_hf_norag_direct_140.json`
   - 結果の記録先として、既存の  
     `B_ベースライン結果テンプレート.md` を再利用するよう指示。

   #### 3-3. FT後モデルの評価コマンド

   - `--hf-lora-path runs/qwen3_law_ft/direct_norag_q8_4bit_v1` を指定したコマンド例を記載。
   - 出力例：`results/evaluations/qwen3_hf_ft_norag_direct_140.json`
   - 結果は `StepI_評価結果_direct.md` などにまとめるよう指示。

   #### 3-4. 比較・考察メモのテンプレ

   - 精度比較（パーセンテージ）
   - 問題ごとの傾向（FT で改善した問題、悪化した問題）
   - RAG や CoT を今後追加するときに注目すべき点  
     など、数項目の見出しだけ用意しておく。

---

## 6. タスクI-4：必要ならコード微修正（最小限）

### 目的

Qwen3:8B HF モデルを heart01 で扱う際に、もし不足しているオプションや  
ハードコードされたパラメータがあれば、**最小限の修正を Codex で行う**。

### やること

1. まずはコードをざっと確認し、以下の点に問題がないかチェックしてください。

   - `scripts/train_qwen_law_ft.py`：
     - 任意の `--model-name` を受け取れるようになっているか。
     - 4bit QLoRA 設定が `use-4bit`, `bnb-4bit-*` で切り替え可能か。
   - `scripts/evaluate_multiple_choice.py`：
     - `--llm-backend hf` / `--hf-model-name` / `--hf-lora-path` オプションがあるか。
   - `app/llm/hf_llm.py`：
     - 4bit ロード（BitsAndBytesConfig）が Qwen3:8B HF モデルにも適用できる設計になっているか。

2. もし不足があれば、**変更箇所を md で整理してから**最小限のコード修正を行ってください。

   - 変更内容は `LLM_ファインチューニング/StepI_コード変更メモ.md` などに記録。
   - 大きな設計変更はこの StepI では行わないでください。

---

## 7. タスクI-5：StepI まとめ md の作成

最後に、現在のチャットで行った作業内容をまとめた md を作成してください。

- 推奨ファイル名：  
  `LLM_ファインチューニング/StepI_まとめ.md`
- 内容例：
  - 作成したファイル一覧
  - Qwen3:8B 本番学習の基本方針
  - heart01 側で次にユーザがやるべきこと（学習 → 評価の順）

---

以上が、**3080 側 Codex に対する StepI（Qwen3:8B 本番学習）指示**です。  
この指示書をもとに、新しい Codex チャット上で StepI 関連の md ファイルと最小限のコード整備を行ってください。