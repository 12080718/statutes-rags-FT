## Step3 タスク2 結果メモ

- 変更ファイル: `scripts/evaluate_multiple_choice.py`
  - プロンプト生成を共通モジュールに統一
    - `app/core/prompts.py` から `build_mc_prompt_direct`, `build_mc_prompt_cot` を import。
    - `create_multiple_choice_prompt` / `create_cot_prompt` をラッパに変更し、内部で共通プロンプトを呼び出す。
  - lawqa形式の選択肢文字列を辞書化する `_parse_choices` を追加（a/b/c/d を小文字で整形）。
  - プロンプトは日本語指示に統一。
- 確認メモ:
  - 簡易なインポート確認を試行したところ、環境の OMP 共有メモリ関連でエラーが発生（`Function Can't open SHM2`）。コード変更自体の構文エラーではなく、環境依存と推測。評価実行前に適宜 `OMP_NUM_THREADS` などを調整してください。

### 次ステップ
- タスク3で `build_finetune_dataset.py` と評価スクリプトの動作確認・サンプル生成を実施する。
