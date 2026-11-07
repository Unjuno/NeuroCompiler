# 再現性と実行手順

## A. 軽量公開版で実行できる確認

Python 3.13.5推奨。再現時はLinux CPU / PyTorch 2.10.0+cpu、NumPy 2.3.5、SciPy 1.17.0を起点とする。その他のStageの依存関係は原本READMEを優先する。

```bash
python3 scripts/verify_manifest.py
python3 -m compileall -q experiments
python3 -m pip install -r requirements-stage9r.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest experiments/stage-09r/tests -q
```

上記は**ソースの整合性とStage9Rの数値単体テスト**であって、全実験のtraining replayではありません。

## B. フル実験の原本を復元

9個の元ZIPを同一フォルダに置き、[`archives/source_archives.csv`](../archives/source_archives.csv)と一致するか検証します。

```bash
python3 scripts/restore_archives.py --archive-dir /path/to/archives --output /tmp/neurocompiler-research-raw
```

展開先には元のアーカイブ構造を一つずつ別フォルダで保存します。パスTraversalを拒否し、既存ファイルは上書きしません。元ZIP自体はGitHub Releases等から取得する運用を想定しています。この公開準備段階では実際のRelease URLは未作成です。

## C. Stageごとの再現範囲

| Stage | 公開版にあるもの | フル原本で追加されるもの | 注意 |
|---|---|---|---|
| 0 | Python、CSV、図、引継ぎ文書 | 特になし | 引継ぎ書のseed表記とコードを監査 |
| 1b–1d | 複数実験スクリプトと集計 | raw task結果 | 1eのコードは回収できていない |
| 1e | report、対照CSV、図 | 原本不明 | 完全再現には実装の復元が必要 |
| 2 | 実験コード/報告・集計 | per-seed raw CSVとログ | calibrationとfinal maskは区別 |
| 3a–3c | 集計・3cスクリプト/レポート | per-seed raw結果 | 3a/3b完全スクリプトは未回収 |
| 4 | semantic/fingerprint試験スクリプトと4d報告 | task単位の結果・ログ | 高周波sin失敗はtarget capacity混入 |
| 5–6 | Tree/paired/rewriteスクリプトと報告 | task単位の結果・ログ | 正規化は+×の対象範囲限定 |
| 7 | ODE/Differential IR, code, checks,報告/図 | bank、モデル重み、raw result | matching失敗、数値精度差を保持 |
| 8 | 非線形ODE・numerical backendのcode/報告 | bank、checkpoint、raw result | 予備発見と最終試験を区別 |
| 9（旧） | 欠落記録のみ | **現存原本なし** | 9Rを再現と称さない |
| 9R | 2D PDEの独立実装、テスト、集計/報告/図 | bank、20seed×4armsのチェックポイントとtask結果 | source-onlyで再学習可、再実行環境依存 |

Stage 9R原本のREADMEに記載された再実行コマンドは以下です。**完走には原本のデータbankと十分なCPU/ストレージが必要**です。

```bash
cd /tmp/neurocompiler-research-raw/stage-09r/neurocompiler_stage9r
python3 -m pytest tests -q
python3 src/experiment.py bank
python3 src/run_all.py
python3 src/evaluate.py
python3 src/refinement.py
python3 src/followup.py
python3 src/replay_check.py
```

`.pt`重みはpickleベースの読み込みを伴うことがあり、信頼できる原本以外をロードしないこと。もしPyTorchのバックエンド・BLAS・CPUが変わるなら丸め差や時間差が出る場合がある。保存済みの時刻合わせDataTimeは別HWでは更新回数も変わるため、原本metadata記録のstep数を使用して再現する。

## D. 数値計算を検証するときの注意

- ODE/PDEの入力座標、残差、誤差は実験内部で無次元化されている。
- 参照解の格子解像度・非線形solver許容差・正規化を変えた結果をそのまま比較しない。
- 計時にはCPU/GPU、クロック、core quota、threads、batch、warmup、反復数、checkpoint読込、IR parseなどの含有区間を記録する。
- 仮にログに同じseedが複数ある場合、結果を独立標本として水増ししない。

## E. CIについて

GitHub Actionsはデフォルトで**静的整合性とPython構文検査**を行います。重い実験本体の再学習は通常のPRで自動実行しません。Stage9Rの9テストはここに記載したローカル環境で実行し合格していますが、全Stageのテスト完走ではありません。
