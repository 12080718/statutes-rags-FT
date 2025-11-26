# Heart01 セットアップ状況報告 (no-RAG中心)

## 実施概要
- リポジトリ: https://github.com/12080718/statutes-rags-FT.git
- 仮想環境: `.venv` 作成 + `pip install -r requirements-llm.txt`（bitsandbytes/accelerate含む）
- HFキャッシュ: `HF_HOME=/home/jovyan/work/.cache/huggingface` で共用パスを設定
- データ: `datasets/lawqa_jp/data/selection.json` 配置済み
- インデックス: RAG用ハイブリッドインデックスは `data/faiss_index_full` → `data/faiss_index` にシンボリックリンクで整合

## 実行結果まとめ
- FTデータ生成(no-RAG direct):
  - コマンド: `scripts/build_finetune_dataset.py --mode direct --no-rag`
  - 出力: `results/finetune/ft_direct_full_norag.jsonl` (140件)
- LoRA学習(no-RAG, HF+4bit, 小モデル1.8B):
  - コマンド: `scripts/train_qwen_law_ft.py --model-name Qwen/Qwen1.5-1.8B-Chat --train-file results/finetune/ft_direct_full_norag.jsonl --output-dir runs/qwen_law_ft/direct_norag_4bit --num-epochs 1 --batch-size 2 --use-4bit --max-steps-override -1 --do-train`
  - 結果: 完走（train_loss ≈ 2.44, 約115秒）
- LoRA適用モデルの評価(no-RAG, HF backend, 同じ1.8B):
  - コマンド: `scripts/evaluate_multiple_choice.py --samples 10 --top-k 1 --no-rag --llm-backend hf --hf-model-name Qwen/Qwen1.5-1.8B-Chat --hf-lora-path runs/qwen_law_ft/direct_norag_4bit`
  - 結果: 完走 (Accuracy 0%, parse unknown 4件) → 出力崩れ/性能低で要改善

## 課題と対応候補
- HF評価で出力崩れ・精度低: 1.8Bモデルの能力不足と生成設定の緩さが要因。
  - 対応案: `--no-few-shot` で簡素化、`LLM_MAX_TOKENS` を小さく設定、temperature=0 などで出力を縛る。より大きなモデルでのLoRA学習をA100で実施する。
- RAG有効時のメモリ/インデックス:
  - Heart01コンテナで埋め込みモデルロード時にOOMのため、RAGなしで検証中。
  - BM25のみ(Retrieverタイプをbm25)でRAG試験、または軽量埋め込みに切り替えて再構築を検討。

## 留意事項
- ベースモデルとLoRA適用モデルは必ず揃える (例: 1.8Bで学習したら評価も1.8B)。
- Ollamaはheart01では非推奨（方針に従いHFバックエンド利用を推奨）。
- カーネル4.18の警告あり: 推奨5.5以上だが現状致命的ではなし。
