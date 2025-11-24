## StepC-1 本番生成プラン（4パターン）

前提: `scripts/build_finetune_dataset.py` を使用し、入力は `datasets/lawqa_jp/data/selection.json`（140問）。few-shot は Step3テストと同じく「有効」を基本とする。

| パターン | use_rag | mode | top_k | --no-rag | --few-shot | 出力例 |
| --- | --- | --- | --- | --- | --- | --- |
| (1) no-RAG + direct | false | direct | 0（無視される） | あり | あり | `results/finetune/ft_direct_full_norag.jsonl` |
| (2) no-RAG + CoT | false | cot | 0（無視される） | あり | あり（CoTでも例は維持） | `results/finetune/ft_cot_full_norag.jsonl` |
| (3) RAGあり top_k=3 + direct | true | direct | 3 | なし | あり | `results/finetune/ft_direct_full_rag_top3.jsonl` |
| (4) RAGあり top_k=3 + CoT | true | cot | 3 | なし | あり（CoTでも例は維持） | `results/finetune/ft_cot_full_rag_top3.jsonl` |

補足:
- no-RAG時の `top_k` は実質無視されるが、0指定で明示。
- few-shotはベースライン/Step3と揃えて「有効」を推奨。必要に応じて `--no-few-shot` でオフにできるが、本計画ではオン。***
