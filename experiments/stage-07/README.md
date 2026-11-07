# NeuroCompiler Stage 7 — 再現用パッケージ

まず `report_ja.md` を読む。数学的導出は `proofs_ja.md`、機械可読の結論は `analysis/contrasts.csv` と `analysis/audit.json` にある。

## 重要な結論

数式→微分IR→ODE解ネットワークの重み生成は動作し、未知微分式の10-step MSEではMetaがDirectを改善した。しかしloss matchingは0/10 seedで不成立。この一次元線形問題では、調整した未学習ランダム特徴とChebyshev法の方が高精度だった。係数外挿時の非有限値を除外していない。

## 内容

- `code/`: 実験、対照、厳密IR、単体テスト、集計コード
- `banks/`: 訓練・validation・開発・最終bank（参照解はFP64）
- `results/confirmatory/seed101`～`seed110`: 保存重み、task結果、matching/LR選択、数値解法、精度・発散対策の追試
- `analysis/`: seed単位統計、MSE集計、失敗件数、再実行照合
- `checks/`: 勾配・微分IR・参照解の検査と環境
- `figures/`: 学習曲線、解法比較、FP32精度劣化、発散対策
- `protocol.json`: 引き継いだ固定条件と再実行であることの記録
- `SHA256SUMS.txt`: ファイル整合性検査

## 実行環境

検証環境: Python3.13.5、PyTorch2.10.0+cpu、NumPy2.3.5、pandas2.2.3、SciPy1.17.0。GPUは使わない。詳細は `checks/environment.json`。CPU上でworker最大3、各worker1スレッド。

既に依存関係がある環境ではそのまま実行できる。必要な場合は次でインストールする。

```bash
python -m pip install -r requirements.txt
```

## 保存結果から統計を再集計

ZIPを展開し、このREADMEのあるディレクトリを作業ディレクトリにする。

```bash
python code/test_differential_ir.py
python code/stage7_experiment.py checks
python code/analyze.py
```

`analyze.py` は非有限値を検出し、failure率を別途出力する。matchingに失敗した比較をPASSとしない。固定task bankを使うため、bootstrapの独立単位はtraining seed。

## 空の別ディレクトリで全学習を再実行

```bash
python code/reproduce.py --output ../stage7_fresh_reproduction
```

指定先が既に存在する場合は停止する。コードと固定データbankをコピーし、10 seedの学習、線形head・強い非学習baseline・backtrackingの追試、集計を実行する。既存重みを読み込んだだけの処理を再学習と呼ばないため、保存結果はコピーしない。OS・BLAS・dtype依存で丸め差や実行時間差が出る可能性がある。報告書の文章は今回の測定のスナップショットであり、再実行時は生成されたCSVを正とする。

## 個別seed

```bash
python code/stage7_experiment.py run --seed 101
python code/linear_head_branch.py --seed 101
python code/basis_controls.py --seed 101
python code/stability_branch.py --seed 101
```

個別実行は同seedの保存結果を更新する。最終テストを見てhyperparameterを調整し直した実験を、同じconfirmatory実験として混ぜない。

## 表記の注意

`DirectBest`はvalidation残差で選んだDirectPhysics。`DirectMatched`はmatching選択を試みた候補であり、今回のmatchingは全seed失敗。`FP32aware`はFP32実行時のvalidation残差でSVD cutoffを選ぶ方式。`Head32`という対照ラベルはFP32実行を指し、当該baselineのcollocation点数は128。`RandomTuned`はニューラル事前学習をしないが、基底スケールとSVD cutoffのvalidation選択を行っている。
