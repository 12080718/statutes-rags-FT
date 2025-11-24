## StepC 生成データサマリ（計画・実行後に更新）

### 想定JSONLファイル一覧（実行後に num_records を追記）

| filename | mode | use_rag | top_k | few_shot | num_records | 備考 |
| --- | --- | --- | --- | --- | --- | --- |
| results/finetune/ft_direct_full_norag.jsonl | direct | false | 0 | true | 140 | no-RAG direct |
| results/finetune/ft_cot_full_norag.jsonl | cot | false | 0 | true | 140 | no-RAG CoT |
| results/finetune/ft_direct_full_rag_top3.jsonl | direct | true | 3 | true | 140 | RAG direct (top3) |
| results/finetune/ft_cot_full_rag_top3.jsonl | cot | true | 3 | true | 140 | RAG CoT (top3) |

### 利用方針（優先度メモ）
- 学習で最優先に使う候補: no-RAG direct / no-RAG CoT（モデルのベース能力強化にシンプル）
- 将来試す候補: RAGあり direct / RAGあり CoT（RAG-awareなFTや耐性検証用）

### 実行後に確認・更新する項目
- 各ファイルの行数（num_records）を `wc -l` で記入
- `meta.use_context`, `meta.top_k`, `meta.retriever_type` が想定どおりかを spot check
- CoT出力が `Reasoning... Answer: <letter>` 形式になっているか
