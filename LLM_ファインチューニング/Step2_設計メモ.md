## タスク1：`prompts.py` の仕様要約

- `_format_choices`
  - 引数: `choices: Dict[str, str]`（キーは `"a"〜"d"` を想定）
  - 戻り値: 各行が `a ...` 形式で並んだ文字列
  - プロンプト構造: 選択肢ブロックを整形する内部ヘルパ（Background/Question/指示文は持たない）

- `build_mc_prompt_direct`
  - 引数: `question: str`, `choices: Dict[str, str>`, `context: Optional[str]=None`, `few_shot: bool=True`
  - 戻り値: 4択用プロンプト文字列（日本語）
  - プロンプト構造:
    - （few_shot時のみ）簡易例1件
    - 【法令条文】`context` またはフォールバック文
    - 【質問】`question`
    - 【選択肢】`_format_choices` で整形した a/b/c/d
    - 【回答指示】条文のみ根拠／回答は1文字／説明不要
    - 出力フォーマット: `回答（1文字のみ）: `
  - 特記: 出力は a/b/c/d の1文字を強制。キーは小文字のみ。

- `build_mc_prompt_cot`
  - 引数: `question: str`, `choices: Dict[str, str>`, `context: Optional[str]=None`, `style: Literal["compact","detailed"]="compact"`
  - 戻り値: CoT用プロンプト文字列（日本語）
  - プロンプト構造:
    - 【法令条文】`context` またはフォールバック文
    - 【質問】`question`
    - 【選択肢】`_format_choices` で整形した a/b/c/d
    - 【推論手順】styleに応じて簡潔/詳細の番号付き手順
    - 出力フォーマット:  
      ```
      Reasoning: <ステップごとの考察>
      Answer: <a/b/c/d の1文字>
      ```

- 仕様整合性コメント
  - 小文字キーa〜dのみを前提 → `_format_choices` が a,b,c,d の順で使用、他は空扱い。
  - `lawqa_jp` の question/choices をそのまま渡す運用 → 問題なし。
  - `context` にRAG条文またはNone → None時はフォールバック文を自動挿入。
  - directモードは1文字のみを要求 → 指示文で明示。
  - CoTモードは Reasoning/Answer 2行構成 → 指示と出力フォーマットで明示。
  - よって、現状仕様で確定して良いと判断。

---

## タスク2：Fine-tuning 用 JSONL フォーマット設計

- `"input"`
  - `prompts.py` で生成したプロンプト文字列（direct か cot かは meta で識別）
- `"output"`
  - directモード: 小文字1文字（例 `"c"`）
  - cotモード: `Reasoning: ...\nAnswer: c` のテキスト
- `"meta"`（学習には使わないが分析用に最低限入れる推奨キー）
  - `id`: データ固有ID
  - `dataset`: 例 `"lawqa_jp"`
  - `mode`: `"direct"` または `"cot"`
  - `correct`: 正解ラベル（小文字1文字）
  - `use_context`: `true/false`（RAGを使ったか）
  - `context_present`: 実際に付与した context が非空か
  - `top_k`: 検索チャンク数
  - `retriever_type`: 例 `"hybrid"` / `"vector"` / `"bm25"`
  - `question`: 元問題文（解析用）
  - `choices`: 選択肢辞書（小文字キー）
  - `source_file`: 元データファイル名や行位置など
  - （任意）`scores` / `doc_ids`: 取得チャンクのIDやスコア配列

### サンプル（directモード）

```json
{
  "input": "あなたは日本の法律に精通したリーガルアシスタントです...回答（1文字のみ）： ",
  "output": "c",
  "meta": {
    "id": "lawqa-0001",
    "dataset": "lawqa_jp",
    "mode": "direct",
    "correct": "c",
    "use_context": true,
    "context_present": true,
    "top_k": 3,
    "retriever_type": "hybrid",
    "question": "〇〇について正しいものはどれか？",
    "choices": {"a": "...", "b": "...", "c": "...", "d": "..."},
    "source_file": "lawqa_jp/selection.json"
  }
}
```

### サンプル（cotモード）

```json
{
  "input": "あなたは日本の法律に精通したリーガルアシスタントです...Answer: <a/b/c/d>",
  "output": "Reasoning: 条文から要件を確認し...\nAnswer: b",
  "meta": {
    "id": "lawqa-0002",
    "dataset": "lawqa_jp",
    "mode": "cot",
    "correct": "b",
    "use_context": false,
    "context_present": false,
    "top_k": 0,
    "retriever_type": "none",
    "question": "〇〇について誤っているものはどれか？",
    "choices": {"a": "...", "b": "...", "c": "...", "d": "..."},
    "source_file": "lawqa_jp/selection.json"
  }
}
```

---

## タスク3：`scripts/build_finetune_dataset.py` の関数設計（インターフェースのみ）

- 想定関数一覧
  - `load_lawqa(path: Union[str, Path]) -> List[Dict[str, Any]]`
  - `build_context(pipeline: RAGPipeline, question: str, top_k: int) -> Tuple[List[Document], str]`
  - `make_instance(sample: Dict[str, Any], context: str, *, mode: Literal["direct","cot"], few_shot: bool=False, top_k: int=0, retriever_type: str="hybrid") -> Dict[str, Any>`
  - `build_dataset(samples: List[Dict[str, Any]], pipeline: Optional[RAGPipeline], *, mode: Literal["direct","cot"]="direct", top_k: int=3, few_shot: bool=False, use_rag: bool=True) -> Iterator[Dict[str, Any]]`
  - `save_jsonl(records: Iterable[Dict[str, Any]], path: Union[str, Path]) -> None`
  - `main() -> None`（CLI: 入力パス、出力パス、mode、top_k、no-rag、few_shot 等）

- 各関数の詳細
  - `load_lawqa`
    - 目的: lawqa_jpのJSON/JSONLを読み込み、`{"question","choices","output",...}` 形式に正規化
    - 引数: `path`（ファイルパス）
    - 戻り値: 正規化済みサンプルのリスト
    - 使用コンポーネント: なし（標準IOのみ）
  - `build_context`
    - 目的: 質問に対してRAGで条文を取得し、`format_context`済みテキストを得る
    - 引数: `pipeline: RAGPipeline`, `question: str`, `top_k: int`
    - 戻り値: `(documents: List[Document], context: str)`
    - 使用コンポーネント: `RAGPipeline.retrieve_documents`, `RAGPipeline.format_context`
  - `make_instance`
    - 目的: 単一サンプル＋コンテキストから JSONL 1レコードを組み立てる
    - 引数: `sample`（lawqa項目）, `context: str`, `mode: "direct"|"cot"`, `few_shot: bool`, `top_k: int`, `retriever_type: str`
    - 戻り値: `{"input": str, "output": str, "meta": {...}}`
    - 使用コンポーネント: `build_mc_prompt_direct` / `build_mc_prompt_cot`
  - `build_dataset`
    - 目的: 全サンプルを処理し、レコード列を生成（top_k件取得しサンプルを複製してもよい設計に拡張可）
    - 引数: `samples`, `pipeline`（RAGを使わない場合Noneでも可）, `mode`, `top_k`, `few_shot`, `use_rag`
    - 戻り値: レコードのイテレータ（生成器）
    - 使用コンポーネント: `build_context`, `make_instance`
  - `save_jsonl`
    - 目的: レコード群をJSONLとして書き出す
    - 引数: `records`（iterable）、`path`
    - 戻り値: なし
    - 使用コンポーネント: なし（標準IOのみ）
  - `main`
    - 目的: CLIエントリ。データ読み込み→RAG初期化→レコード生成→保存の一連処理
    - 引数: なし（argparseで受け取る）
    - 戻り値: なし
    - 使用コンポーネント: `load_config`, `RAGPipeline`, 上記各関数

- 処理フロー（lawqa_jp → RAG → JSONL）
  1. CLI引数を受け取り、入力データパス・出力パス・mode（direct/cot）・top_k・few_shot・no-rag 等を決定。
  2. `load_lawqa` でサンプルを読み込む。
  3. RAGを使う場合は `load_config` と `RAGPipeline` を初期化（no-rag時は `pipeline=None`）。
  4. 各サンプルについて `build_context`（use_rag=Trueなら）で条文を取得し、context文字列を作る。
  5. `make_instance` で `prompts.py` の関数を用いて `input` を生成し、`output` と `meta` を組み立てる。
  6. `build_dataset` がレコードを順次yieldする（必要なら top_k件のドキュメントで複数インスタンス展開）。
  7. `save_jsonl` でレコードをJSONLに書き出す。
  8. 完了メッセージと簡易統計（件数など）を表示。
