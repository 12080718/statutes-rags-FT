# StepH RTX3080用・小モデルデバッグ指示（Codex用）

RTX3080（10GB）の環境では、7B/8Bクラスのモデルを QLoRA で学習させると CUDA OOM が発生しやすいです。  
そこで、この StepH では **「小さい HF モデル（1.5〜4B級）」を使って、学習〜評価パイプラインが正しく動くかをデバッグ** します。

ここでは **コードの改造は原則行わず**、既に実装済みのスクリプトを使って、

- `train_qwen_law_ft.py` による LoRA 学習（小モデル）
- `evaluate_multiple_choice.py` + HF バックエンドによる評価（小モデル）

が一通り通ることを確認してください。

最終的な本番実験（Qwen3:8B での FT）は、A100 20GB 環境で行う前提です。

---

## 共通ルール

- **ブランチ**  
  既に実装自体は完了しているため、新しい機能追加はしません。  
  コード変更が必要な場合のみ、新しいブランチ（例：`feature/steph-small-model-debug`）を切って作業してください。

- **GPU リソース**  
  - この StepH は RTX3080 (10GB) 前提です。
  - CUDA OOM が出た場合は、**設定やモデルサイズを下げて再トライ**してください。  
    コードの大幅変更は行わないでください。

- **重い実験禁止**  
  - このステップでは、**長時間の学習・140問フル評価**は行わないでください。
  - 目的はあくまで「**パイプラインが正しく動くかの確認**」です。

- **記録**  
  - 実行したコマンド・結果の要約は  
    `LLM_ファインチューニング/StepH_結果まとめ.md`  
    に必ず追記してください。

---

## 事前前提（既に存在しているもの）

以下のスクリプト・設定が StepF / StepG までで実装済みである前提で進めてください。

- `scripts/train_qwen_law_ft.py`
  - `--use-4bit`, `--bnb-4bit-compute-dtype`, `--bnb-4bit-quant-type` オプションあり
  - no-RAG 用 FT データ（direct / CoT）を学習できる実装
- `scripts/evaluate_multiple_choice.py`
  - `--llm-backend`, `--hf-model-name`, `--hf-load-in-4bit` オプションあり
- `results/finetune/ft_direct_full_norag.jsonl`
- `results/finetune/ft_cot_full_norag.jsonl`
- HF バックエンド向けラッパ：
  - `app/llm/hf_llm.py`（`HFLLMConfig`, `HFLoRALLM`）
- RAG 設定：
  - `app/core/rag_config.py`
  - `app/retrieval/rag_pipeline.py`

---

## タスクH-1：デバッグ用の「小さい HF モデル」を決める

### 目的

RTX3080 で QLoRA 学習と HF バックエンド評価をテストするための  
**小さいチャットモデル（1.5〜4B級）**を一つ決めます。

### やってほしいこと

1. HF のモデル名は、仮に以下のようなものを**第一候補**としてください（実在モデル名は環境に合わせて調整）。

   - 候補例1: `Qwen/Qwen1.5-1.8B-Chat`
   - 候補例2: `Qwen/Qwen1.5-4B-Chat`

2. 実際に使用できるモデル名（HF から取得可能なもの）を 1つ選び、

   - `選んだモデル名`
   - `パラメータサイズ（おおよそ）`

   を `LLM_ファインチューニング/StepH_結果まとめ.md` にメモしてください。

3. もし上記の候補が利用できない場合は、

   - 代わりに使ったモデル名
   - 選定理由（例：「手元にすでにダウンロード済みだったため」など）

   を同様に記録してください。

---

## タスクH-2：no-RAG direct 用 小モデル学習のスモールラン

### 目的

デバッグ用の小モデルを使い、**no-RAG direct 用 FT データ**で  
`train_qwen_law_ft.py` の学習ループが通るか確認します。

### やってほしいこと

1. 以下のようなコマンドを、**選んだ小モデル名に置き換えて**実行してください。

   ```bash
   python scripts/train_qwen_law_ft.py      --model-name <小モデルのHFモデル名>      --train-file results/finetune/ft_direct_full_norag.jsonl      --output-dir runs/debug_small_model/direct_norag_small_4bit      --num-epochs 1      --batch-size 2      --learning-rate 2e-4      --max-seq-length 512      --gradient-accumulation-steps 2      --warmup-ratio 0.03      --lora-r 16      --lora-alpha 32      --lora-dropout 0.05      --use-4bit      --bnb-4bit-compute-dtype bfloat16      --bnb-4bit-quant-type nf4      --do-train
   ```

   - **重要**：ここでは
     - エポック数は `1`
     - `max-seq-length` は `512` 程度
     - バッチサイズも小さめ  
     とし、**VRAM負荷を抑えたスモールラン**にしてください。
   - もし `--max-steps-override` オプションがある場合は、  
     それを使って「数百ステップ程度」で打ち切って構いません。

2. 実行中に CUDA OOM が出た場合：

   - まずは
     - `max-seq-length` をさらに小さくする（例: 384, 256）
     - `batch-size` を 1 に落とす
   - それでもダメな場合は、
     - `--use-4bit` を外して fp16/bfloat16 モードで試す  
       （その場合、学習はかなり遅くなりますが、パイプライン検証目的なら許容範囲です）

3. 学習が最後まで完了したら、以下を確認してください。

   - `runs/debug_small_model/direct_norag_small_4bit` 以下に
     - LoRA 重み
     - Trainer のログ・チェックポイント
     が生成されていること。

4. 実行に使った**最終的なコマンド**と、

   - 学習完了の有無（成功 / OOM / その他エラー）
   - 実行時間のおおよその目安
   - 生成されたディレクトリ / ファイルのパス

   を `LLM_ファインチューニング/StepH_結果まとめ.md` に記録してください。

---

## タスクH-3：HF バックエンド + 小モデルでの評価スモーク

### 目的

タスクH-2で学習した小モデル（LoRA）を HF バックエンド経由で呼び出し、  
`evaluate_multiple_choice.py` が **HF モードで動くか**を確認します。

### やってほしいこと

1. `evaluate_multiple_choice.py` を HF バックエンドモードで、  
   **2〜3問だけ** 評価してください。

   - 例（モデル名・LoRA パスは環境に合わせて変更）：

   ```bash
   python scripts/evaluate_multiple_choice.py      --data datasets/lawqa_jp/data/selection.json      --output results/evaluations/small_model_debug_direct.json      --samples 3      --top-k 1      --no-rag      --llm-backend hf      --hf-model-name <タスクH-2で使用した小モデル名>      --hf-lora-path runs/debug_small_model/direct_norag_small_4bit      --hf-load-in-4bit      --ensemble 1
   ```

   ポイント：

   - まずは **no-RAG direct** で十分です。
   - RAG 付きや CoT モードのテストは、余力があれば追加で行ってください。

2. 実行時に以下を確認してください。

   - `Initializing RAG pipeline...` の後に
     - `RAG Mode: Disabled (LLM only)`（`--no-rag` の場合）
     - `LLM Model: <小モデル名>`
     - `Backend: hf`（ログに表示されていれば）
   - `EVALUATION RESULTS` が表示され、
     - `Total Samples: 3`
     - `Accuracy: ...`  
     が出力されること。
   - `results/evaluations/small_model_debug_direct.json` が生成されていること。

3. CUDA OOM や import エラーが出た場合は：

   - そのエラーメッセージを `StepH_結果まとめ.md` に貼り付け
   - 原因と思われるポイントを簡単にメモしてください。

4. 正常に完走した場合は、

   - 使用したコマンド
   - HF バックエンドでのログの要点（backend, model_name, lora_path など）
   - 精度と簡単な感想（例：「3問中2問正解」など）

   を `LLM_ファインチューニング/StepH_結果まとめ.md` に記録してください。

---

## タスクH-4：StepH 全体の振り返りメモ

### 目的

A100 での本番 Qwen3:8B 学習に進む前に、  
3080 上での小モデルデバッグで得られた知見を整理します。

### やってほしいこと

1. `LLM_ファインチューニング/StepH_結果まとめ.md` の最後に、  
   「StepHまとめ」セクションを追加してください。

2. そこに、次の観点で箇条書きしてください。

   - 使用した小モデル名と、そのサイズ感（1.8B / 4B など）
   - 4bit QLoRA 学習が RTX3080 でどこまで問題なく動いたか
     - （例：OOM なし / 設定調整で回避可能 / どうしても OOM 等）
   - HF バックエンド評価の挙動
     - direct / no-RAG の小規模テストが通ったか
   - A100 20GB での本番 Qwen3:8B 学習に向けて、
     - 「そのまま流用できそうな設定」
     - 「A100 では変えた方が良さそうな設定（例：batch-size, max-seq-length 等）」

3. これらのメモは、次の Step（A100 上での本番 FT & 評価）で  
   設定を決める際の参考情報として使います。

---

以上が StepH（RTX3080用・小モデルデバッグ）の指示内容です。  
この StepH では **実験パイプラインが 3080 上で正しく動くこと** を確認することが目的であり、  
最終的な精度や大規模実験は A100 環境で行います。