# 再起動後の環境復元クイックガイド

このドキュメントは、コンテナ/ホスト再起動後に statutes-rags を再利用するための最小手順をまとめたものです。

## 前提
- 初回セットアップ（uv/venv作成、Ollamaセットアップ、データ/インデックス生成）が完了していること
- `.env` は更新済みで、`VECTOR_STORE_PATH=data/faiss_index_full` を指していること

## 手順（毎回これだけ）
```bash
cd ~/yokote/statutes-rags-master
source setup/restore_env.sh
```
- uv / venv への PATH を復元
- Python 仮想環境を有効化
- Ollama を起動（既に起動中ならスキップ）
- SudachiPy などのトークナイザー設定を適用

## 便利な確認コマンド
```bash
which python                   # .venv 配下になっているか
python -V                      # バージョン確認
echo $VECTOR_STORE_PATH        # data/faiss_index_full になっているか
curl -s http://localhost:11434/api/tags | head -40  # Ollama 応答確認
```

## よく使う後続コマンド
```bash
# クイック評価（10件）
./scripts/evaluate.sh 10

# 50件評価（10〜15分目安）
./scripts/evaluate.sh 50

# 結果の確認
ls -lh results/evaluations | tail
cat results/evaluations/evaluation_results_final.json | python3 -m json.tool | head -40
```

## トラブル時のヒント
- `Permission denied` でシェルスクリプトが実行できない場合: `chmod +x scripts/evaluate.sh` または `bash scripts/evaluate.sh ...`
- Ollama が応答しない場合: `pkill ollama && (cd setup && ./bin/ollama serve > ollama.log 2>&1 &)` を試す
- インデックスが見つからないと言われた場合: `.env` の `VECTOR_STORE_PATH` が `data/faiss_index_full` になっているか確認
