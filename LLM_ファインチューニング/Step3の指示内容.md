# Step3の指示内容（実装フェーズ）

これから Step3（実装フェーズ）を進めたいです。  
基本方針・設計は以下に書かれている内容に従います。

- `docs/今後の方針.md`
- `LLM_ファインチューニング/Step2の指示内容.md`
- `LLM_ファインチューニング/Step2の結果（設計メモ）`
- 実装済みの `app/core/prompts.py`

---

## 共通ルール（必ず守ってほしいこと）

- 作業は **新しいブランチ上** で行ってください（例：`feature/llm-ft-impl`）。  
  `main` を直接変更しないでください。
- 変更は **できるだけ小さく分割** してください。
  - 例：  
    - 「build_finetune_dataset.py を追加」→ 1コミット  
    - 「evaluate_multiple_choice.py から prompts.py を使うように変更」→ 1コミット
- 破壊的な操作は行わないでください。
  - 例：大量のファイル削除、プロジェクト構成の大幅な変更、`pip uninstall` など
- 何か大きく設計を変えたくなった場合は、  
  **先に Markdown で提案を書いてから**にしてください（このステップでは実装しないでOK）。
- すべてのタスクについて、最後に **「どのファイルがどう変わったか」** を簡単に Markdown でまとめてください。

---

## タスク1：`scripts/build_finetune_dataset.py` の実装

### 目的

Step2 で設計したインターフェースに基づき、  
**Fine-tuning 用 JSONL データを生成するスクリプト**を実装します。

- 入力：`lawqa_jp` 形式の4択問題（selection.json など）
- 出力：`{"input": ..., "output": ..., "meta": {...}}` 形式の JSONL ファイル
- RAG を使う / 使わない を切り替え可能にする

### 実装上の制約

- 新規ファイル **`scripts/build_finetune_dataset.py`** を作成してください。
- 既存ファイルは、ここではまだ **`evaluate_multiple_choice.py` 以外は編集しない** でください（必要になったら後述のタスクで扱います）。
- lawqa_jp の読み方は、現状の `scripts/evaluate_multiple_choice.py` の実装に合わせて構成してください  
  （同じ `selection.json` 形式を扱えるようにする）。

### 必要な関数（Step2 設計を元に実装）

以下の関数を実装してください。名前・役割は Step2 で設計したものに合わせます。

1. `load_lawqa(path: Union[str, Path]) -> List[Dict[str, Any]]`
   - 目的：
     - lawqa_jp の selection.json（など）を読み込み、
       ```python
       {
         "id": str,
         "question": str,
         "choices": {"a": str, "b": str, "c": str, "d": str},
         "correct": str,  # "a"〜"d" の小文字
       }
       ```
       のような共通フォーマットに正規化する。
   - 実装メモ：
     - 可能であれば `evaluate_multiple_choice.py` の既存処理を参考にする。
     - 必要なら「TODO: 後で共通化」コメントを残す。

2. `build_context(pipeline: RAGPipeline, question: str, top_k: int) -> Tuple[List[Document], str]`
   - 目的：
     - 指定された `question` に対して、RAG で条文チャンクを `top_k` 件取得し、  
       `RAGPipeline.format_context` と同等の形でコンテキスト文字列を生成する。
   - 戻り値：
     - `documents`: `List[Document]`
     - `context`: str（条文を連結したテキスト）

3. `make_instance(sample: Dict[str, Any], context: str, *, mode: Literal["direct","cot"], few_shot: bool = False, top_k: int = 0, retriever_type: str = "hybrid") -> Dict[str, Any]`
   - 目的：
     - 1問分のサンプル＋コンテキストから、  
       Fine-tuning 用 JSONL の1レコードを作る。
   - 処理内容：
     - `sample` から `question`, `choices`, `correct` を取得。
     - `mode` に応じて：
       - `direct` → `build_mc_prompt_direct` を呼んで `input` を作る／`output` は `"c"` など1文字のみ。
       - `cot` → `build_mc_prompt_cot` を呼んで `input` を作る／`output` は `Reasoning... Answer: c` 形式。
     - `meta` には Step2 で決めた最低限のキーを詰める：
       - `id`, `dataset`, `mode`, `correct`, `use_context`, `top_k`, `retriever_type`, `question`, `choices`, `source_file` など。

4. `build_dataset(samples: List[Dict[str, Any]], pipeline: Optional[RAGPipeline], *, mode: Literal["direct","cot"] = "direct", top_k: int = 3, few_shot: bool = False, use_rag: bool = True) -> Iterator[Dict[str, Any]]`
   - 目的：
     - 全サンプルについて `make_instance` を呼び出し、  
       Fine-tuning 用レコードを順に生成する。
   - 挙動（初期バージョン）：
     - `use_rag = True` かつ `pipeline` がある場合：
       - `build_context` で context を作る（現時点では「top_k 件まとめて1 context」としてOK）。
     - `use_rag = False` の場合：
       - `context = ""` または `None` を渡す（prompts 側のフォールバックに任せる）。
     - 1問につき **1レコード** 生成（top_kごとに増やす拡張は後回し）。

5. `save_jsonl(records: Iterable[Dict[str, Any]], path: Union[str, Path]) -> None`
   - 目的：
     - `{"input": ..., "output": ..., "meta": ...}` 形式の dict を 1行1 JSON として書き出す。

6. `main() -> None`
   - 目的：
     - CLI エントリポイント。
   - 想定する CLI 引数（argparse などで実装）：
     - `--lawqa-path`（入力データ）
     - `--output-path`（出力 JSONL ファイル）
     - `--mode` (`direct` or `cot`)
     - `--top-k`（context 用の取得件数）
     - `--use-rag / --no-rag`
     - `--few-shot`（direct/cotの few_shot を有効にするか）
   - フロー：
     1. 引数をパース。
     2. `load_lawqa` で samples を読み込み。
     3. `use_rag` が true の場合：
        - `load_config()` から設定を読み、`RAGPipeline` を初期化。
     4. `build_dataset` でレコード列を生成。
     5. `save_jsonl` で JSONL を書き出し。
     6. 件数などの簡単な統計を stdout に出す。

### タスク1の最後にやってほしいこと

- 変更ファイル一覧（少なくとも `scripts/build_finetune_dataset.py`）と、  
  各ファイルの変更内容を Markdown で簡単にまとめてください（`LLM_ファインチューニング/Step3_タスク1_結果.md` などに書いてOK）。

---

## タスク2：`evaluate_multiple_choice.py` から `prompts.py` を使うように変更

### 目的

既存の 4択評価スクリプトが、  
**`app/core/prompts.py` で定義したプロンプト関数を使うように統一**します。

### 実装上の制約

- 編集対象はまず `scripts/evaluate_multiple_choice.py` のみとし、  
  他のファイルの大きな変更は行わないでください。
- 現状の CLI オプションや結果の出力形式は **変えない** でください。
  - ※ 必要なら「内部的に使うプロンプト生成関数だけ差し替える」イメージ。

### やってほしいこと

1. `app/core/prompts.py` から以下を import してください。
   - `build_mc_prompt_direct`
   - `build_mc_prompt_cot`

2. `evaluate_multiple_choice.py` 内で、  
   現在 4択プロンプトを作っている関数（例：`create_multiple_choice_prompt`, `create_cot_prompt`）を調べてください。

3. それらを以下のようにリファクタしてください。
   - direct 用プロンプト生成 → 中身を `build_mc_prompt_direct` 呼び出しに置き換える。
   - CoT 用プロンプト生成 → `build_mc_prompt_cot` 呼び出しに置き換える。
   - 必要なら wrapper 関数として残しても構いません（評価スクリプト内の他コードとの互換性を保つため）。

4. 既存の `extract_answer()` や評価ロジックと整合が取れているか確認してください。
   - direct:
     - 出力は 1文字の `"a"〜"d"` を前提にパース。
   - CoT:
     - `Answer: c` の形式から 1文字を抜き出す（既存実装を確認し、必要なら正規表現などを微調整）。

### 軽い動作確認（可能な範囲で）

- 小さなサンプル（5問程度）で `evaluate_multiple_choice.py` を動かし、  
  例外なく動作することを確認してください（実行方法は既存 README 等に従う）。
- 精度はこの時点では気にしなくてよいです。  
  ※ 目的は「プロンプトの置き換えによって評価スクリプトが壊れていないこと」の確認です。

### タスク2の最後にやってほしいこと

- どの関数をどのように `prompts.py` に置き換えたか、  
  変更前後の関係を Markdown で簡単にまとめてください  
  （例：`LLM_ファインチューニング/Step3_タスク2_結果.md`）。

---

## タスク3：簡単なエンドツーエンド確認とメモ作成

### 目的

- 実装した `build_finetune_dataset.py` が実際に JSONL を生成できるか確認する。
- base-line の 4択評価が、prompts 統一後も問題なく走るか確認する。
- この結果を、後で自分（人間）が見返せるようにメモしておく。

### やってほしいこと

1. **データ生成のテスト**
   - lawqa_jp の selection.json（など）から、  
     10問程度（小さいサンプル）を対象に `build_finetune_dataset.py` を実行し、
     - direct モード用 JSONL（例：`ft_direct_sample.jsonl`）
     - cot モード用 JSONL（例：`ft_cot_sample.jsonl`）
     を出力してください。
   - 生成された JSONL の先頭数行を表示し、  
     - `"input" / "output" / "meta"` が想定通りになっているか  
     - `mode`, `correct`, `use_context` などが妥当か  
     を確認してください。

2. **評価スクリプトのテスト**
   - 既存の `evaluate_multiple_choice.py` を、元々使っていたのと同等の設定で少数問だけ動かし、  
     例外が出ず、結果が出力されることを確認してください。
   - できれば、prompts 差し替え前と結果（正答率）が大きく変わっていないことも確認してください  
     （多少の変動はOKです）。

3. **結果メモの作成**
   - `LLM_ファインチューニング/Step3_まとめ.md` のようなファイルを作成し、以下をまとめてください。
     - 実装したファイルと主な関数
     - `build_finetune_dataset.py` の使い方（CLI 例）
     - サンプル JSONL の例（1レコード分）
     - 評価スクリプトの簡単な実行例と結果（正答率だけでOK）

---

以上のタスクを、この順番で実行してください。  
各タスクの完了時には、**何を変更したか・どんな確認をしたか**を必ず Markdown で残してください。
