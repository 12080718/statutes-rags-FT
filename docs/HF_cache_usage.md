# HFキャッシュの設定と活用メモ

## なぜ設定するか
- Hugging Face のモデル/埋め込みをダウンロードした際のキャッシュ場所を明示し、再ダウンロードを防ぐ。
- コンテナ環境でメモリ負荷やI/O負荷を下げ、OOMやダウンロード失敗を回避しやすくする。

## 共用キャッシュパスの例
```
export HF_HOME=/home/jovyan/work/.cache/huggingface
```
- 毎回シェルで `export` するか、`~/.bashrc` や `.env` に記述しておくと再利用できる。

## 先にモデル/埋め込みをキャッシュしておく方法
- 埋め込みモデルの例（軽量版に切り替える場合はモデル名を変更）
```
export HF_HOME=/home/jovyan/work/.cache/huggingface
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large')"
```
- 以降の実行で同じモデルを参照する際は、キャッシュから読み込まれるためダウンロード不要。

## 実行時の例（evaluate_multiple_choice）
```
export HF_HOME=/home/jovyan/work/.cache/huggingface
python scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection.json \
  --output results/evaluations/quick_check.json \
  --samples 3 \
  --top-k 1 \
  --llm-backend ollama  # または hf
```
- 埋め込みモデルを小さめにしたい場合は `.env` の `EMBEDDING_MODEL` を `intfloat/multilingual-e5-base` などに変更するとメモリ負荷を軽減できる。

## 注意
- キャッシュを置くパスは書き込み可能な永続ディレクトリにする（例: `/home/jovyan/work/.cache/huggingface`）。
- 既に別パスにキャッシュがある場合は、`HF_HOME` を揃えて使い回すと再ダウンロードを避けられる。***
