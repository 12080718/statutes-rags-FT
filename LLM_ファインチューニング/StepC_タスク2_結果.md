## StepC-2 no-RAG 本番データ生成（提案のみ、未実行）

対象: lawqa_jp selection（140問）、コンテキストなしでFT用JSONLを生成する想定。few-shotはStep3テストと同様「有効」を基本とする。

### コマンド案
1) no-RAG + direct
```
python scripts/build_finetune_dataset.py \
  --lawqa-path datasets/lawqa_jp/data/selection.json \
  --output-path results/finetune/ft_direct_full_norag.jsonl \
  --mode direct \
  --no-rag \
  --top-k 0 \
  --few-shot
```

2) no-RAG + CoT
```
python scripts/build_finetune_dataset.py \
  --lawqa-path datasets/lawqa_jp/data/selection.json \
  --output-path results/finetune/ft_cot_full_norag.jsonl \
  --mode cot \
  --no-rag \
  --top-k 0 \
  --few-shot
```

### 実行後の確認手順（人間が実施）
- `wc -l results/finetune/ft_direct_full_norag.jsonl`（想定: 140行）
- `wc -l results/finetune/ft_cot_full_norag.jsonl`（想定: 140行）
- `head -n 3 <file>` で `input/output/meta` を目視。特に:
  - `meta.mode` が direct / cot
  - `meta.correct` が a〜d
  - `meta.use_context` が false
  - `meta.top_k` が 0
  - `meta.retriever_type` が none
  - CoTの場合、outputが `Reasoning... Answer: <letter>` 形式になっているか

※ 上記コマンドは提案のみで、実行はしていません。
