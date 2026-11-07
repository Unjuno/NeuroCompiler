# NeuroCompiler

**構造化された数学的知識を、ニューラルネットワークのパラメータまたは適切な数値計算へ変換する研究プロジェクト。**

[English overview](README.en.md) · [研究史](docs/RESEARCH_HISTORY.md) · [実験結果一覧](docs/RESULTS_AT_A_GLANCE.md) · [ロードマップ](docs/ROADMAP.md) · [再現手順](docs/REPRODUCIBILITY.md)

> **研究段階のプロトタイプです。** 汎用的な数学理解や、従来のPDE数値法に対する優位は実証されていません。Stage 9Rは独立再実装であり、会話中に報告された旧Stage 9の再現ではありません。すべての数値を「対象・比較手法・評価量・実験seed数」の範囲で解釈してください。

## 研究の目的

「形式知識を、対応する問題を解くのに利用できるモデル状態へ変換できるか」を検証します。初期構想は **Knowledge → Weights** でしたが、実験から次の分業が妥当と分かりました。

```text
数式・演算子・初期/境界条件・問題パラメータ
                  │
                  ▼
   Symbolic parser / canonical semantic IR
                  │
                  ▼
       Backend selection / validation
         ┌────────┴─────────┐
         ▼                  ▼
 Neural weight compiler   Numerical solver
 (DirectFit / MetaInit)   (Galerkin / Newton / Spectral)
         └────────┬─────────┘
                  ▼
  解・残差評価・必要に応じた数値補正
```

ニューラル生成器は**常に最良の選択ではありません**。物理制約と強い非学習対照を入れたStage 7–9Rは、この点を明確にしています。

## 主な実験結果と限界

| 範囲 | 報告済み結果（すべて限定条件付き） | 重要な限界 |
|---|---|---|
| Stage 0–2 | 未見の加法compositionで、10-step MetaInitが初期誤差を揃えたDirectFitより相対的に高速適応 | DirectFit最終モデルはzero-shot/絶対MSEでなお強い |
| Stage 3 | Generic DeepSetsの一般化は失敗。加法則を明示したtoken方式では改善 | Dense MLPと等価な関数クラスであり汎用的な数式理解ではない |
| Stage 4 | Gaussianの未見 `(center,width)` descriptorから重み生成が可能 | 既知の平滑な1関数族内の一般化 |
| Stage 5–6 | AST演算子/木構造は重要。`+`,`×` の特定同値rewriteは正規化で不変化 | 長期Meta優位は未支持。一般の数学的同値性は保証しない |
| Stage 7–8 | ODE残差学習・短期適応を検証。強い古典法は高精度 | 数値法の優位を保持。訓練と推論コストは別に比較する |
| Stage 9R | 二次元非線形PDEでは物理損失の同更新数改善は小さい。同時間では教師学習が優位 | 36基底の限界と係数生成誤差を分離。従来Galerkin法が高精度 |

統計量、区間、seed数、対照条件、失敗例は [結果一覧](docs/RESULTS_AT_A_GLANCE.md) と各stageの原報告書で参照してください。

## リポジトリ構造

```text
NeuroCompiler/
├── README.md / README.en.md
├── experiments/
│   ├── stage-00/ ... stage-08/
│   ├── stage-01e/          # 独立した構造OOD追試
│   ├── stage-09/           # 旧Stage9の原本不足を記録
│   └── stage-09r/          # 保存済みの独立再実装
├── docs/                  # 概念、研究史、結果、ロードマップ、監査方針
├── archives/              # オリジナルZIPのチェックサムと来歴
├── scripts/               # マニフェスト検証・安全な原本展開
├── .github/workflows/     # コード/整合性の最低限のCI
└── requirements-stage9r.txt
```

`experiments/` のコードは**歴史的な実験コードをできる限り原形保持したもの**です。共通APIへ無理に統合しておらず、すべてのstageがこの軽量版だけで完全に再実行可能という意味ではありません。

## 最低限の動作確認

```bash
python3 scripts/verify_manifest.py
python3 -m pip install -r requirements-stage9r.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -m pytest experiments/stage-09r/tests -q
```

Stage 9Rの大規模再学習には原本ZIP内の問題bankと再現環境が必要です。詳細は [再現手順](docs/REPRODUCIBILITY.md) を参照してください。保存された `.pt` ファイルは信頼できる出所を確認しない限り読み込まないでください。

## 公開方針・原本

このGitHub用の軽量版には、主要な論文用報告書、実験コード、集計CSV、図を収録します。**学習checkpoint（`.pt`）、大きなtask bank（`.npz`）、task単位の生CSVは原本ZIPにのみ保持**しています。原本ZIPをGitHubリポジトリのgit履歴へ直接コミットしないでください。Stage 9Rの原本ZIPは単体で100MBを超えます。GitHub Releases等に別添する設計です。

[原本ZIPのハッシュ索引](archives/source_archives.csv) · [格納方針](docs/PUBLIC_RELEASE.md) · [証拠の区分](docs/EVIDENCE_POLICY.md)

**原本Archiveの公開状況：GitHub Releases未登録（2026-11-08）。** [検証済みRelease候補・SHA-256](docs/ARCHIVE_RELEASE_CANDIDATE.md) · [公開の最終手順](docs/RELEASE_OPERATIONS.md) · [公開前の安全性監査](docs/ARCHIVE_SECURITY_REVIEW.md)

### ライセンスについて

リポジトリの公開状態（Public）は再利用許諾を意味しません。著作権者・共同研究者・使用データ/コードの権利関係が未確認のため、この公開準備版に**確定LICENSEは含めていません**。使用許諾を選択してから初回公開してください。`LICENSE.template` は選択時の注意書きです。

## 次の研究課題

1. **同じ精度・実時間・前処理費用での比較**：ニューラル生成器が数値バックエンドを上回る問題族を定義する。
2. **構造OODと表現能力の分離**：有限基底誤差と係数生成誤差を別々に測定する。
3. **検証可能なsolver router**：残差に基づく採用判断、数値補正、fallbackを条件付きで行う。

[反証可能なStage 10ロードマップ](docs/ROADMAP.md)

---

**状態（2026-11-07）**：過去に作成された9つの原本ZIPは整合性検査済み。旧Stage 9は原本がなく会話由来の履歴だけを保存し、Stage 9Rを別実験として区別しています。外部査読や他者による独立再現を意味するものではありません。
