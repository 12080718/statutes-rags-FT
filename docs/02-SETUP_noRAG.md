# statutes-rags セットアップ（RAGなし版の最短手順）

RAGを使わずにLLMのみで評価/学習するための簡易セットアップ手順です。インデックス構築や埋め込みモデルの準備は不要です。

## 0. 前提
- Python 3.10 以上
- GPU推奨（ローカル: 3080クラス、heart01: A100/MIG）
- リポジトリ: `https://github.com/12080718/statutes-rags-FT.git`
- データ: `datasets/lawqa_jp/data/selection.json` を配置済みであること

## 1. 取得と仮想環境
```
git clone https://github.com/12080718/statutes-rags-FT.git
cd statutes-rags-FT
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-llm.txt
```
※ HFモデルを使う場合は `bitsandbytes`/`accelerate` も requirements に含まれています。

## 2. （任意）HFキャッシュ設定
```
export HF_HOME=/home/あなたのユーザ/.cache/huggingface  # 書き込み可能なパス
```
モデル/埋め込みのダウンロードを共有キャッシュに保存して再利用します。

## 3. 評価（no-RAG）
Ollamaを使う場合の例（サーバ起動済み/モデルpull済みを前提）:
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection.json \
  --output results/evaluations/quick_check_no_rag.json \
  --samples 3 \
  --top-k 1 \
  --no-rag \
  --llm-backend ollama \
  --llm-model qwen3:8b
```
HFバックエンドを使う場合の例:
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection.json \
  --output results/evaluations/quick_check_no_rag_hf.json \
  --samples 3 \
  --top-k 1 \
  --no-rag \
  --llm-backend hf \
  --hf-model-name Qwen/Qwen1.5-1.8B-Chat  # 学習と同じベースモデルに揃える
```
※ RAGを使わないため `--no-rag` を必須指定。インデックスは不要です。

## 4. Fine-tuning 用データ生成（no-RAG）
```
python scripts/build_finetune_dataset.py \
  --lawqa-path datasets/lawqa_jp/data/selection.json \
  --output-path results/finetune/ft_direct_full_norag.jsonl \
  --mode direct \
  --no-rag
```
（CoT版は `--mode cot --no-rag` に変更）

## 5. LoRA学習（HF+4bit例、軽量設定に調整可）
```
python scripts/train_qwen_law_ft.py \
  --model-name Qwen/Qwen1.5-1.8B-Chat \
  --train-file results/finetune/ft_direct_full_norag.jsonl \
  --output-dir runs/qwen_law_ft/direct_norag_4bit \
  --num-epochs 1 \
  --batch-size 2 \
  --use-4bit \
  --max-steps-override -1 \
  --do-train
```
※ GPUメモリに合わせて batch/seq-length を調整。4bitが厳しい場合は `--use-4bit` を外し fp16/bf16 で実行。

## 6. LoRA適用モデルの評価（no-RAG, HFバックエンド）
```
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection.json \
  --output results/evaluations/hf_ft_direct_no_rag.json \
  --samples 10 \
  --top-k 1 \
  --no-rag \
  --llm-backend hf \
  --hf-model-name Qwen/Qwen1.5-1.8B-Chat \
  --hf-lora-path runs/qwen_law_ft/direct_norag_4bit
```

## 7. トラブルシュート（no-RAG想定）
- `ModuleNotFoundError`: `.venv` が有効か確認し、`pip install -r requirements-llm.txt` を再実行。
- HFモデルのダウンロードで落ちる: `HF_HOME` を設定し、キャッシュを共有。小型モデルを指定する。
- Ollama接続エラー: サーバ起動/モデルpull状態を確認するか、HFバックエンドで評価する。***
- LoRAロードエラー（サイズ不一致）: 学習と同じベースモデルを評価でも指定する（例: 1.8Bで学習したら評価も `--hf-model-name Qwen/Qwen1.5-1.8B-Chat`）。
