# statutes-rags セットアップガイド（ローカル/heart01 共通の最小手順）

本書は、現行構成（HFバックエンド前提、Ollamaは任意）でのセットアップ手順を簡潔にまとめたものです。

## 前提
- Python 3.10 以上
- GPU推奨（ローカル: RTX3080 クラス、heart01: A100 20GB/MIG）
- GitHubリポジトリ: `https://github.com/12080718/statutes-rags-FT.git`

## 1. リポジトリ取得
```
git clone https://github.com/12080718/statutes-rags-FT.git
cd statutes-rags-FT
```

## 2. Python仮想環境と依存インストール
推奨: `.venv` を作成し、依存を一括インストール（HFバックエンド必須、Ollamaは不要）。
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-llm.txt
# 4bit/QLoRAを使う場合は bitsandbytes, accelerate も requirements に含まれています
```

## 3. 環境変数（必要に応じて）
`.env` を作成して上書きできます（デフォルトは `app/core/rag_config.py` の通り）。
主要項目:
- `LLM_BACKEND` (`ollama`/`hf`、デフォルト `ollama`)
- `LLM_MODEL` (Ollama用、例 `qwen3:8b`)
- `HF_MODEL_NAME` (HFバックエンド用、例 `Qwen/Qwen1.5-7B-Chat`)
- `LORA_PATH` (LoRA適用時のパス)

## 4. データセット配置
`datasets/` 配下に以下を用意（リポジトリには含まれません）:
- `datasets/egov_laws/*.xml`（e-Gov法令XML）
- `datasets/lawqa_jp/data/selection.json`（4択評価用）

## 5. インデックス構築（RAGを使う場合のみ）
ベクトル/BM25/ハイブリッドのインデックスを作成:
```
make index
# もしくは scripts/build_index.py を直接実行
```

## 6. 動作確認
- RAG評価（少数問）:
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection.json \
  --output results/evaluations/quick_check.json \
  --samples 3 \
  --top-k 1 \
  --llm-backend ollama  # または hf
```
- HFバックエンドを使う場合は `--llm-backend hf --hf-model-name <HFモデル名>` を指定。

## 7. Fine-tuning データ生成
```
python scripts/build_finetune_dataset.py \
  --lawqa-path datasets/lawqa_jp/data/selection.json \
  --output-path results/finetune/ft_direct_full_norag.jsonl \
  --mode direct --no-rag
```
CoTやRAG版は `--mode cot` / `--top-k 3` などで切り替え。

## 8. Fine-tuning 学習（HF+LoRA）
ローカルでは小モデル（1.5〜4B）でトライ、heart01では Qwen3:8B で本番を実行:
```
python scripts/train_qwen_law_ft.py \
  --model-name Qwen/Qwen1.5-1.8B-Chat \
  --train-file results/finetune/ft_direct_full_norag.jsonl \
  --output-dir runs/qwen_law_ft/direct_norag_4bit \
  --num-epochs 1 \
  --batch-size 2 \
  --use-4bit \
  --do-train
```
※ 実行前にGPUメモリに合わせて batch/seq-length を調整すること。

## 9. Fine-tuned モデルの評価（HFバックエンド）
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection.json \
  --output results/evaluations/hf_ft_direct.json \
  --top-k 3 \
  --llm-backend hf \
  --hf-model-name Qwen/Qwen1.5-7B-Chat \
  --hf-lora-path runs/qwen_law_ft/direct_norag_4bit \
  --use-cot   # CoT評価する場合
```

## 10. heart01 での利用メモ
- Codexは使えないため、ローカルで準備したコードをGitHub経由でpullして利用。
- HFバックエンドのみ使用（Ollamaは不要/非推奨）。
- 依存は `.venv` で `pip install -r requirements-llm.txt`。
- データ/インデックスは事前にコピーまたは再構築。

## トラブルシューティング
- GPUメモリ不足: 小さいモデルや batch/seq-length 縮小、またはCPU実行。
- 4bit関連エラー: bitsandbytesが未インストール/非対応の可能性。`pip install bitsandbytes` と環境を確認。
- デバイス不一致: HFバックエンドは入力をモデルデバイスへ移動する実装済み。依然出る場合は `--llm-backend hf --hf-load-in-4bit` を確認し、CPU実行も検討。
