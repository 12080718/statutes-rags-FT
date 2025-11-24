## Step3 まとめ

### 実装・変更
- `scripts/build_finetune_dataset.py`
  - lawqa_jp 4択問題から FT 用 JSONL (`{"input","output","meta"}`) を生成するスクリプトを実装。
  - 主な関数: `load_lawqa` / `build_context` / `make_instance` / `build_dataset` / `save_jsonl` / `create_retriever` / `main`。
  - 依存ライブラリ未導入環境でも `--no-rag` で動くよう、RAG依存を遅延インポートに変更。
- `scripts/evaluate_multiple_choice.py`
  - プロンプト生成を `app/core/prompts.py` の共通関数に差し替え（`build_mc_prompt_direct` / `build_mc_prompt_cot`）。選択肢を a/b/c/d 辞書に揃える `_parse_choices` を追加。

### データ生成テスト
- direct モード（no-rag, few-shot, 10件）
  - コマンド:  
    `python scripts/build_finetune_dataset.py --lawqa-path datasets/lawqa_jp/data/selection.json --output-path results/finetune/ft_direct_sample.jsonl --mode direct --top-k 3 --samples 10 --no-rag --few-shot`
  - 生成件数: 10
  - 例（1件目抜粋）:  
    - input: 日本語指示＋few-shot例＋質問/選択肢  
    - output: `"c"`  
    - meta: `{id, dataset, mode=direct, correct=c, use_context=false, top_k=0, retriever_type=none, question, choices, source_file}`
- cot モード（no-rag, 5件）
  - コマンド:  
    `python scripts/build_finetune_dataset.py --lawqa-path datasets/lawqa_jp/data/selection.json --output-path results/finetune/ft_cot_sample.jsonl --mode cot --top-k 3 --samples 5 --no-rag`
  - 生成件数: 5
  - 例（1件目抜粋）:  
    - input: 日本語CoT指示＋質問/選択肢  
    - output: `Reasoning: ...\nAnswer: c`  
    - meta: `{id, dataset, mode=cot, correct=c, use_context=false, top_k=0, retriever_type=none, question, choices, source_file}`

### 評価スクリプト動作確認
- 依存の LLM/Ollama および langchain-community が環境に無いため、`evaluate_multiple_choice.py` の実行は未実施。LLM環境が整い次第、少数サンプルでの動作確認を推奨。
- `build_finetune_dataset.py` は `--no-rag` なら追加依存なしで実行可能。

### 次のアクション候補
1. LLM/Ollama 環境を整えた上で、評価スクリプトを少数サンプルで実行し、prompts差し替え後の動作を確認。
2. RAGありでの JSONL 生成（`--top-k`指定、リトリーバ依存を満たした環境で）を試す。
3. 生成した JSONL を用いた LoRA/FT 学習パイプラインに接続し、学習・評価を実行。
