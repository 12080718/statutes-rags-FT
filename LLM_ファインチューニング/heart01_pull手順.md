## ローカル編集をGitHubへ反映し、heart01で取得する流れ

1. ローカル(3080)で変更確認
   ```
   git status
   ```
2. 変更ファイルをステージング
   ```
   git add <ファイル>   # 全部なら git add .
   ```
3. コミット
   ```
   git commit -m "edit ..."
   ```
4. リモートにプッシュ
   ```
   git push
   ```
   これでGitHubに反映される。

5. heart01側で更新を取得
   ```
   git pull
   ```
   （既存クローンが無ければ `git clone https://github.com/12080718/statutes-rags-FT.git`）

補足:
- heart01ではHFバックエンドのみ利用。依存は `.venv` を作って `pip install -r requirements-llm.txt` などで揃える。***
