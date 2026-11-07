# 研究史 — Stage 0からStage 9Rまで

**証拠の基本規則**：数値はそれぞれの保存済みレポート・集計CSVから転記した*条件付きの報告値*です。今回リポジトリを整理する作業は、すべての学習を初期状態から独立再実行するものではありません。比較対象、評価量、seed数、失敗例を必ず同時に読む必要があります。

| 段階 | 研究の問い | 結果・重要な発見 | 解釈と制約 | 参照 |
|---|---|---|---|---|
| **0** | 3 primitiveの未知和から重みを生成できるか | HyperInitの0更新MSE中央値0.460、FO-MAML0.916、Random0.939 | zero-shotの優位。高速適応そのものは未証明 | [原引継ぎ](../experiments/stage-00/knowledge_to_weights_research_handoff.md) |
| **1b/1d** | zero-shot品質と適応性を分離できるか | h=2は差+0.0015 CIが0を跨ぐ。h=10のloss-matched Stage1dは差−0.7394 CIが負 | meta horizon重要。最終DirectFitは0更新で強い | [報告](../experiments/stage-01/stage1_experimental_report.md) |
| **1e** | calibrationと別の未見compositionへ転移するか | h=10平均差−0.6889、95% CI[−0.9414,−0.4573]、n=20 | h=20はloss matching 3/20のみで比較不適切 | [報告](../experiments/stage-01e/stage1e_report.md) |
| **2a/b/c** | 8 primitiveへ拡張し構造・係数OODを検証 | 2aの差−0.3086 CI[−0.3930,−0.2287]、n=20。2bの5mask平均−0.5578、n=10 | relative improvementであってabsolute MSE最良とは限らない | [報告](../experiments/stage-02/stage2_final_report.md) |
| **3a** | generic DeepSetsでtoken入力化 | 差−0.0564 CI[−0.2204,+0.1169]、n=10 | **FAIL/不確実**。cardinality/未見compositionに弱い | [集計](../experiments/stage-03/results/stage3a_contrasts.csv) |
| **3b/3c** | 加法則に合わせたtoken encoder・初期化公平化 | Stage3cの差−0.4283 CI[−0.5596,−0.3035]、n=10 | Denseと関数クラス等価。順序不変性は数値的に確認 | [報告](../experiments/stage-03/stage3c_report.md) |
| **4 pilot** | 既知descriptorから未見の高次sin/cosを外挿 | 高周波sin/cosで性能が崩れた | tanh targetの近似/最適化限界が混入したためsemantic encodingのみの否定不可 | [pilot](../experiments/stage-04/docs/stage4_pilot_report.md) |
| **4d** | Gaussian `(mu,sigma)` の未見identityを扱えるか | Semantic Meta差−0.1539、n=10。zero-shot semantic/ID-only MSE幾何比0.1422 | **既知Gaussian族内**の意味座標一般化 | [報告](../experiments/stage-04/stage4d_report.md) |
| **5a/b/d** | `+`,`×` AST構造とMetaの短期利得 | Tree/Bagのzero-shot MSE幾何比0.385、n=10。Stage5d early-AUC差−0.0602 CIが負 | Stage5b 100-step全域は差−0.0251 CIが0を跨ぐ | [報告](../experiments/stage-05/stage5_stage6_report.md) |
| **6/6b** | 数式の書き換えで重みが変わる問題 | Syntax Treeにはrewrite感度。canonical front-endでテストした同値式の重み差0 | **設計で強制した不変性**。演算は+×限定、5 seed pilot | [報告](../experiments/stage-05/stage5_stage6_report.md) |
| **7** | 微分記号をODEに適用し、物理残差から調整する | Metaは10更新後Directに改善傾向。しかし線形解法とChebyshev法がはるかに高精度 | Stage7の初期loss matchingは0/10。FP32変換劣化も発見 | [報告](../experiments/stage-07/report_ja.md) |
| **8** | 非線形ODEなら学習済み生成が有利か | 低budget MetaはDirectより劣り、強いChebyshev/定数初期値+Newtonが高精度 | source配置と精度を切り分けた。solver routing仮説が浮上 | [報告](../experiments/stage-08/report_ja.md) |
| **9（旧）** | 二次元・変係数PDEへ移行 | 会話では予備成果が報告された | **原本未確認**。旧Stage9の数値を検証済みとして載せない | [欠落記録](../experiments/stage-09/README.md) |
| **9R** | 物理原理を損失と応答に導入し学習費用を揃える | 位置OODでData104.46%、Response101.66%、DataTime98.88%、Galerkin36=3.05%。係数生成が主誤差 | 独立再実装・20seed×4arms。Stage9旧版の再現ではない | [報告](../experiments/stage-09r/report_ja.md) |

## 研究上の転換点

**0→1**: 良いzero-shotと高速adaptationを区別。2-stepでは適応効果なし。

**2→3**: 入力表現をtoken化するだけでは不十分。合成則のinductive biasを入れるが、それだけで汎用的な意味の学習を示したわけではない。

**4→6**: 未知primitiveの意味descriptorと、AST演算構造を導入。数学的に保証できる同値性は記号front-endに委ねる。

**7→9R**: 強い非学習数値法との比較により、「重み生成できた」だけでは実用価値が証明されないことが明確に。ソルバー選択・補正・誤差監査を研究対象へ追加。

## 結果を引用するときの禁止事項

- 相対log-AUCと絶対MSEを同じ尺度として比較しない。
- `n=200 tasks ×10 seeds` を `n=2000 independent runs` と書かない。独立単位は原則training seed。
- 10% loss-matchedの失敗例を消さない。
- 同値式への不変性を「NNが数学を理解した」証拠にしない。
- 9Rを旧Stage9の再現として扱わない。
- 既存数値法より精度が悪い結果を省かない。

研究の現在位置、改善/反証計画は [ロードマップ](ROADMAP.md) に分離しました。
