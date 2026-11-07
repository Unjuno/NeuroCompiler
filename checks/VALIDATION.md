# 公開用整理時の確認記録（2026-11-07）

このファイルは公開用索引・保存済みコードの整合性チェックの結果を記載します。**過去の全モデルを今回再学習・再現した記録ではありません。**

- 9個の原本ZIP：`ZipFile.testzip()` でCRCエラー無し、各原本についてSHA256計算済み。
- 公開用の研究原報告書、集計CSV、ソース、図、補足sidecarのコピーを確認。原パスとSHA256を `archives/curated_source_map.csv` に記録。
- Stage9Rの既存pytest suite：**9件PASS**（Python3.13.5、PyTorch2.10.0+cpu、NumPy2.3.5、SciPy1.17.0）。これ以外のstageの訓練は今回未実行。
- `python -m compileall -q experiments scripts`：成功（Python構文のみの確認）。
- `scripts/check_repository.py`：整理用Markdownで85件の相対リンクを確認し、不在0件（歴史的報告書が外部rawに言及するリンクは範囲外）。
- `scripts/reconcile_metrics.py`：Stage1e/Stage2a/Stage3a/Stage9Rの6件の著しい数値転記を元CSVと照合し、**6/6 PASS**。これは統計検定の独立再計算ではない。
- Public用ファイルのテキストに対して一般的なAPIキー・秘密鍵の表層パターンを走査。該当したメールアドレス様パターンはPython行列積（`@`演算子）の誤検知と確認。全面的な機密監査・法務監査ではない。
- 古いStage9の実行コード・生データが現存ファイル群に無いことを記録。Stage9Rを独立再実装として明示。

### 読み方

**ファイルの検証と研究結論の独立再現は別物**です。ここでのPASSはソース索引と保存済み実装の限定された検査を示します。論文の結果を第三者が再現した、という主張はしていません。
