# heart01 → GitHub → 3080 反映手順（LoRA出力も含める場合）

## 前提
- heart01 側で `.venv` 等はコミットしない。
- LoRA出力やチェックポイントをプッシュする場合、リポジトリ容量が増えるので注意（必要なら git-lfs 利用を検討）。
- リモート名/ブランチは運用に合わせて読み替え（例: origin/main）。

## heart01 側：コミット＆プッシュ
1. 作業ディレクトリへ移動し状態確認  
   `git status`
2. 不要な大容量がステージされていないか確認（例: `.venv/`, `datasets/`, `logs/`）。LoRA出力を含めたい場合は `runs/..` の対象フォルダのみ選んで add。  
3. 変更をステージ  
   `git add <必要なファイル/ディレクトリ>`  
   例: `git add scripts/evaluate_multiple_choice.py scripts/split_selection_dataset.py LLM_ファインチューニング/*.md runs/qwen3_law_ft/direct_v2_norag_q8_4bit_v1`
4. コミット  
   `git commit -m "Add FT v2 artifacts and instructions"`  
5. プッシュ  
   `git push origin <ブランチ名>`

## 3080 側：取得
1. リポジトリディレクトリへ移動  
2. 最新を取得  
   `git pull origin <ブランチ名>`
3. 仮想環境を有効化（必要なら）  
   `source .venv/bin/activate`  
   依存追加があれば `pip install -r requirements-llm.txt`

## 備考
- LoRAやcheckpointを頻繁にプッシュする場合は容量肥大に注意し、不要になった古いチェックポイントは別途整理する。***
