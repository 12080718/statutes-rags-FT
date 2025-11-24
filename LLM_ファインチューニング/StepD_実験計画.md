## StepD-1 Fine-tuning 実験パターン案

| パターン | 使用JSONL | 目的 | 評価方法案 |
| --- | --- | --- | --- |
| A: no-RAG direct | `results/finetune/ft_direct_full_norag.jsonl` | 1文字回答専用モデル。RAGなしLLMのベース精度向上。 | `evaluate_multiple_choice.py` を directモード（--no-cot）で実行。 |
| B: no-RAG CoT | `results/finetune/ft_cot_full_norag.jsonl` | Reasoning+Answer 出力の品質向上。 | `--use-cot` でCoT評価。Answer抽出精度を見る。 |
| C: no-RAG direct+CoT 混合 | 両方を結合 or シャッフルしたJSONL（要事前マージ） | multi-taskで direct/CoT 両対応モデルを狙う。 | direct評価 + CoT評価の両方で確認。 |
| D: RAG-aware (direct/CoT) | `ft_direct_full_rag_top3.jsonl` / `ft_cot_full_rag_top3.jsonl` | 取得文脈を含む入力での挙動最適化。RAG-awareな回答を狙う。 | RAG有効で direct / CoT 評価。meta.use_context=true を前提。 |

優先度メモ: まず A, B を優先（no-RAGベースを固める）。次に C（混合）を検討。RAG-aware の D は余力があれば。
