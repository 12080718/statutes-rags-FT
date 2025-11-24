## StepC-3 RAGあり本番データ生成（提案のみ、未実行）

対象: lawqa_jp selection（140問）、RAG top_k=3でFT用JSONLを生成する想定。few-shotはStep3テストと同様「有効」を基本とする。

### コマンド案
3) RAGあり + direct
```
python scripts/build_finetune_dataset.py \
  --lawqa-path datasets/lawqa_jp/data/selection.json \
  --output-path results/finetune/ft_direct_full_rag_top3.jsonl \
  --mode direct \
  --top-k 3 \
  --few-shot
```

4) RAGあり + CoT
```
python scripts/build_finetune_dataset.py \
  --lawqa-path datasets/lawqa_jp/data/selection.json \
  --output-path results/finetune/ft_cot_full_rag_top3.jsonl \
  --mode cot \
  --top-k 3 \
  --few-shot
```

### 実行後の確認手順（人間が実施）
- 行数確認: `wc -l results/finetune/ft_direct_full_rag_top3.jsonl` / `wc -l ..._cot...`（想定: 140行）
- `head -n 3 <file>` で `input/output/meta` を目視。特に:
  - `meta.use_context` が true
  - `meta.top_k` が 3
  - `meta.retriever_type` が retriever名（vector/bm25/hybrid 等）
  - direct: `output` が小文字1文字
  - cot: `output` が `Reasoning... Answer: <letter>` 形式
  - `input` の context 部分に条文テキストが含まれているか

※ 上記コマンドは提案のみで、実行はしていません。
