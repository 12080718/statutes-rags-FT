# StepCの指示内容（Fine-tuning 用データの本番生成）

これから StepC（Fine-tuning 用データの本番生成）を進めたいです。  
基本方針・設計は以下に書かれている内容に従います。

- `docs/今後の方針.md`
- `LLM_ファインチューニング/Step2の指示内容.md`
- `LLM_ファインチューニング/Step2の結果（設計メモ）`
- `LLM_ファインチューニング/Step3_タスク1_結果.md`
- `LLM_ファインチューニング/Step3_タスク2_結果.md`
- `LLM_ファインチューニング/Step3_まとめ.md`
- `LLM_ファインチューニング/A_環境要件まとめ.md`
- `LLM_ファインチューニング/B_ベースライン実行コマンド_本番140問.md`
- 実装済みの `scripts/build_finetune_dataset.py` と `app/core/prompts.py`

---

## 共通ルール

- 作業は **既に使っている作業ブランチ** か、新しいブランチ（例：`feature/llm-ft-data`）上で行ってください。  
  `main` を直接変更しないでください。
- コードの大きな変更はこのステップでは行いません。  
  （`build_finetune_dataset.py` と `prompts.py` の仕様前提で「使う」フェーズです）
- **重いコマンド（本番データ生成）は「コマンドを提示 → 実行は人間が行う」という前提**にしてください。  
  ここでは、実行例と確認手順を Markdown にまとめるところまでで構いません。
- 各タスクが終わったら、必ず  
  `LLM_ファインチューニング/StepC_タスクX_結果.md`  
  のような Markdown に「何をしたか・どんなファイルができたか」を簡潔にメモしてください。

---

## タスクC-1：本番生成プランとファイル命名の整理

### 目的

Fine-tuning 用データを **どのパターンで何本作るか** を先に整理し、  
ファイル名・オプションを決めておきます。

### やってほしいこと

1. 今までの結果（Bタスクまで）を踏まえ、次の4パターンのデータを「候補」として整理してください。

   - (1) no-RAG + direct（CoTなし）
   - (2) no-RAG + CoT
   - (3) RAGあり（top_k=3）+ direct
   - (4) RAGあり（top_k=3）+ CoT

2. 各パターンについて、

   - 使用スクリプト：`scripts/build_finetune_dataset.py`
   - 入力ファイル：`datasets/lawqa_jp/data/selection.json`（140問）
   - 想定出力ファイル名例（提案）：
     - `results/finetune/ft_direct_full_norag.jsonl`
     - `results/finetune/ft_cot_full_norag.jsonl`
     - `results/finetune/ft_direct_full_rag_top3.jsonl`
     - `results/finetune/ft_cot_full_rag_top3.jsonl`
   - 使用するオプションの組み合わせ：
     - `--mode direct|cot`
     - `--top-k`
     - `--no-rag` の有無
     - `--few-shot` を使うかどうか  
       （基本は Step3 テストと同じ設定でよいですが、理由があれば変更案をコメントしてください）

   を Markdown の表形式で整理してください。

3. この表を  
   `LLM_ファインチューニング/StepC_タスク1_計画.md`  
   として保存してください。

---

## タスクC-2：no-RAG 本番データ生成（direct / CoT）

### 目的

RAG を使わない **純粋な LLM 入力用 FT データ**を、本番サイズ（全140問）で生成します。

### やってほしいこと

1. タスクC-1で決めた方針に従って、次の2つのコマンド例を提示してください。

   - (1) no-RAG + direct 用 JSONL 生成
     - 例：
       ```bash
       python scripts/build_finetune_dataset.py \
         --lawqa-path datasets/lawqa_jp/data/selection.json \
         --output-path results/finetune/ft_direct_full_norag.jsonl \
         --mode direct \
         --no-rag \
         --top-k 0 \
         --few-shot
       ```
   - (2) no-RAG + CoT 用 JSONL 生成
     - 例：
       ```bash
       python scripts/build_finetune_dataset.py \
         --lawqa-path datasets/lawqa_jp/data/selection.json \
         --output-path results/finetune/ft_cot_full_norag.jsonl \
         --mode cot \
         --no-rag \
         --top-k 0
       ```

   - 実際のオプション値（`--few-shot` を付けるか、`--top-k` をどうするか）は、Step3 のテストや StepB のベースライン設定と整合するように提案してください。

2. 上記コマンドを **「人間が実行する想定」**で書いてください。  
   このドキュメント内ではコマンドを提示するだけで、実際の実行は行わないでください。

3. コマンド実行後にやるべき確認手順も箇条書きで書いてください。

   例：

   - `wc -l` で行数を確認（想定行数 ≒ サンプル数 or サンプル数×パターン数）。
   - `head -n 3` で `input/output/meta` の構造を目視確認。
   - `meta.mode`, `meta.correct`, `meta.use_context` が意図通りになっているか。

4. これらのコマンド例と確認手順を  
   `LLM_ファインチューニング/StepC_タスク2_結果.md`  
   に Markdown でまとめてください。

> ※ 実際のコマンド実行は人間が行います。  
>  実行後に、必要ならこのファイルに「実際の行数」「気づいた点」を追記しても構いません。

---

## タスクC-3：RAGあり本番データ生成（direct / CoT）

### 目的

RAG（top_k=3）を使った場合の Fine-tuning 用データを、本番サイズで生成します。  
将来的に「RAG-aware な FT」を試すためのデータです。

### やってほしいこと

1. タスクC-1の計画に従って、次の2つのコマンド例を提示してください。

   - (3) RAGあり + direct 用 JSONL 生成
     - 例：
       ```bash
       python scripts/build_finetune_dataset.py \
         --lawqa-path datasets/lawqa_jp/data/selection.json \
         --output-path results/finetune/ft_direct_full_rag_top3.jsonl \
         --mode direct \
         --top-k 3
       ```
   - (4) RAGあり + CoT 用 JSONL 生成
     - 例：
       ```bash
       python scripts/build_finetune_dataset.py \
         --lawqa-path datasets/lawqa_jp/data/selection.json \
         --output-path results/finetune/ft_cot_full_rag_top3.jsonl \
         --mode cot \
         --top-k 3
       ```

   - `--few-shot` を付けるかどうかは、StepB までのベースライン設定との一貫性を考えて提案してください。

2. no-RAG のときと同様に、実行後に確認すべきポイントを箇条書きで書いてください。

   例：

   - `meta.use_context` が `true` になっているか。
   - `meta.top_k` が 3 になっているか。
   - `input` の `context` 部分に条文テキストが含まれているか（何行か目視）。

3. これらを  
   `LLM_ファインチューニング/StepC_タスク3_結果.md`  
   に Markdown としてまとめてください。

---

## タスクC-4：生成データのサマリと注意点の整理

### 目的

生成された JSONL ファイル群を一覧にし、  
あとで学習スクリプトを組む際に迷わないように「メタ情報」を整理します。

### やってほしいこと

1. タスクC-2, C-3 で生成する想定の JSONL ファイルを、表形式で整理してください。

   - 列の例：
     - `filename`
     - `mode`（direct / cot）
     - `use_rag`（true/false）
     - `top_k`
     - `few_shot`（true/false）
     - `num_records`（実際に生成した行数を後から記入する欄）
     - 備考（例：学習候補か、予備データか）

2. 上記の表とともに、

   - 「学習で最優先に使う候補」（例：no-RAG direct / no-RAG CoT）
   - 「将来試す候補」（例：RAGありバージョン）

   を簡単にコメントしてください。

3. これらを  
   `LLM_ファインチューニング/StepC_まとめ.md`  
   に Markdown でまとめてください。

> ※ 実際の `num_records` は、ユーザーがコマンドを実行したあとで埋めればOKです。  
>  このタスクでは「表の枠」と「どう使い分けるかのコメント」まで作成してください。

---

以上のタスクを、この順番で実行してください。  
各タスクの完了時には、

- どのコマンドを提案したか
- どの JSONL ファイルを作る想定か

を、指定した Markdown ファイルに必ず残してください。
