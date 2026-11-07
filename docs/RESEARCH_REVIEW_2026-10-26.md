# NeuroCompiler — 独立技術評価と反証実験ロードマップ

評価日：2026-10-26

## 判定

**研究プロトタイプとして有望。ただし2026年時点の先行研究を踏まえると、NeuroCompilerの広義のアイデアだけでは新規性が不足する。性能優位・計算経済性・意味保存の独自の組合せを、より難しい未見問題で実証する必要がある。**

本レビューは会話に提供されたStage 1–6の報告、マウント済みStage 7のレポート・実装・保存CSV、2026-10-26時点で確認した一次論文を基にした。Stage 8の確定済みCSV・ソースコードは今回の閲覧可能なファイル群に存在しないため、途中進捗は確証と扱わない。

## 成立していること（経験的事実・限定条件）

- Stage 5の同じleaf・係数で演算子配置だけを変えたテストで、構造認識Treeが構造無視Bagのzero-shot MSEより優位。10 training seeds; seed-level mean log ratio -0.9544, CI [-0.9719,-0.9393]（Stage 5/6レポート）。
- Stage 5dの別の事前固定short-horizon評価でMetaはloss-matched DirectFitに対し平均early-log-AUC差 -0.06020, seed bootstrap CI [-0.09366,-0.02755]。効果は最初の1–10更新に集中。
- Stage 6bのsymbolic canonicalizationで、演算の結合・交換・分配の同値rewriteから同じcanonical representationを作り、5 seedsの対象ペアで生成weight差0。これは**ニューラルネットが意味を独力で学習した証拠ではなく、front-endの決定論的性質**。
- Stage 7の微分IRは有理数係数・加法・乗法・微分で正規化可能、実装上の最大展開項数4096と最大次数32に制限。1200の同値ペアで一致し、400の非同値例で誤同一化なし（テスト集合に限る）。
- Stage 7の一次線形ODE u'+λu=f, u(-1)=bで、未知右辺ASTに対する10 SGD更新後の解MSE比較：Meta / validation-best DirectPhysics のseed-level mean log ratio -0.320181、95% CI [-0.477912,-0.135578]、8/10 seedsでMeta有利。
- Stage 7の初期validation正規化残差matchingは0/10 seedsで成立しなかった。**初期lossを制御したoptimization geometryの機構比較としては無効**。
- Stage 7の高精度基準では、同じstructure tasksでMetaの学習済み特徴＋線形最小二乗（FP32 aware）MSE=1.905e-7、調整済みランダム特徴＋最小二乗=1.832e-12、Chebyshev FP32=9.677e-16。強い古典・非学習ベースラインへの優位はない。
- λ外挿で固定LRの非有限値発生率：DirectBest 26.81%、Meta 22.50%、RegMeta 20.44%（各160 task×10 seed）。support残差に基づくbacktrackingでは観測対象に関して0件。一般保証ではない。
- Stage 7のtargetは1→32 tanh→1、97 weights。一方、初期値 b は `u=b+(x+1)N_θ(x)` の別定数として渡す。**仕様全体が97 weightsへ埋め込まれるのではない**。

## 結論を過大解釈すべきでない点

1. Stage 7で左辺演算子はずっと一次線形演算子 `u'+λu`。未見は主に右辺ASTおよび葉・係数パラメータであり、未知の任意ODE演算子・非線形PDEの転移とは異なる。
2. 神経表現のパラメータ非同定性（hidden-neuron置換等）を評価する際、weight差を直接機能差と同一視しない。
3. 学習seedに関するCIは固定task bankに条件付けられている。task分布の多様性、設計選択、文法族一般化などは包含しない。
4. 強いbaselineとSGDでは点数・forward/linear solveコストが一致しないため、MSEの勝敗だけから計算時間の勝敗を宣言しない。
5. Symbolic equivalenceの保証は定義したIRの代数法則・有理数計算範囲に限られる。任意の解析的同値を判断する一般oracleではない。
6. Stage 7は以前の実験の保存データ・ソースを用いた**再実行**であって、別研究室・別task bankでの独立再現ではない。

## 新規性を制約する重要な先行研究

- Ha et al., HyperNetworks (ICLR 2017), arXiv:1609.09106 — 別のNNの重みを生成する発想。
- Finn et al., MAML (ICML 2017), arXiv:1703.03400 — 少数gradient stepに適した初期化。
- Belbute-Peres et al., HyperPINN (2021), arXiv:2111.01008 — 方程式パラメータからPINN重みを生成する。
- Dwivedi & Srinivasan, PIELM (2019), arXiv:1907.03507 — 物理残差とランダム特徴の高速線形解法。
- Ye et al., PDEformer-1 (2024), arXiv:2407.06664 — PDEの記号・数値情報を計算グラフで表現し、zero-shot / fine-tuneを行う。
- Ye et al., PDEformer-2 (2025), arXiv:2507.15409 — 2Dの複数PDEとAST様の計算グラフ入力。
- Liu et al., KAN 2.0 / Kolmogorov-Arnold Networks Meet Science (Physical Review X, 2025), DOI:10.1103/4t7t-v19l — kanpilerは数式をKANに直接compileし、微調整も可能。
- Bischof et al., HyPINO (NeurIPS 2025), arXiv:2509.05117 — hypernetworkからPDE用PINN重みを作り、zero-shotおよびfine-tuningを比較。

**差別化の候補は** `rewrite-verified differential IR + compact deployable parameter state + explicitly short-horizon optimized initialization + uncertainty/precision contract` という組合せ。ただし、この組合せそのものの完全新規性は文献総覧なしに断言できない。

## 数学上の要点

有理数係数の有効なIR正規化 `C` と決定論的重み生成器 `G` について、`C(e₁)=C(e₂)` なら

`G(C(e₁))=G(C(e₂))`。

この保証は構文上の同値rewriteに対する表現不変性であり、生成NNの実際の解精度を保証するものではない。逆向きの命題（生成weightが同じなら元式が同値）は一般に成立しない。

### 大量task処理の償却条件

- N: タスク件数（無次元の非負整数）
- T_train: 生成器を事前学習する総時間（秒）
- t_comp: taskあたりNeuroCompilerの入力前処理・重み生成・必要な適応・実行を含む時間（秒/task）
- t_baseline: taskあたりの最良非学習solverの同精度到達時間（秒/task）

学習器の総費用 `T_train+N*t_comp` がsolverの総費用 `N*t_baseline` より小さくなる必要十分条件は `t_baseline>t_comp` のとき

`N > T_train / (t_baseline-t_comp)`。

時間単位は右辺で秒/(秒/task)=task、両辺で一致。これを満たせない場合は「同精度での時間優位」は立証できない。速度だけでなくメモリ、保存モデル転送、FLOPs、失敗率を別軸で記録する。

## 提案ロードマップ（優先順位順）

### Gate 0: Stage 8の実測完結

- 非線形ODE `u'+λu+βu³=f` の10独立training seeds、固定 untouched task bank、非有限値を含む完全報告。
- 振幅・剛性・境界条件に沿った失敗分析。
- 精度と時間の両方で強いChebyshev・adaptive integrator・最適化済みrandom featuresと比較。
- **STOP**: 低次元非線形でも古典法が優位なら、解法性能を主な新規性に据えない。

### Gate 1: 機構の因果分離

- 初期 **解MSE** と残差の双方をmatching。成功seedだけで結論を出さず、matching成功率を報告。
- DirectFit、MAML/Reptile、HyPINO風hypernetwork、Random/tuned random、固定重み＋解法変更を比較。
- Seed≥20、複数独立task bank、paired hierarchical bootstrap、事前固定primary。

### Gate 2: 公平なend-to-end Pareto

- 同じ誤差閾値へ到達するwall-clock、エネルギー、メモリ、前処理時間、generator推論、inner steps、失敗率。
- offline train費用を含めたamortization break-evenを算出。
- task分布が変わっても閾値を維持できるか独立評価。

### Gate 3: 真の差別化

- 演算子そのもの（非線形・非可換・変係数）、境界型、次元、領域形状をtrain/testで完全holdout。
- 記号的意味保存・ユニット整合性・数値安定性契約を評価。
- HyPINO、PDEformer、FNO/DeepONet、kanpiler、古典数値法との比較。

## H/T/D/C/U

H: 意味保存front-endを持つNeuroCompilerは、訓練外の方程式族でも、指定された解誤差しきい値に対し、学習コストを償却可能な問題件数の範囲で最良baselineより早く解を提供できる。

T: 先験的にtask familyと精度しきい値を固定し、独立task bankとtraining seed≥20、同じHW/dtype/thread/精度検証で比較する。

D: primaryは同精度到達時間のpaired log ratioの95%区間上端<0、非有限結果は失敗として計上。条件を満たさないならFAIL/UNCERTAIN。別のprimaryとしてrewrite同値ペアの100%一致と異義ペアの識別率を検証。

C: 学習済み重みは高速でなく、適切な古典数値法やrandom featuresでも十分に表現・解ける。

U: training seed、task-family sampling、precision/cutoff、generator外挿、線形系条件数、HWおよび実行系の非決定性。総合不確かさは今後推定。

## 最終判定

研究を中止する理由はない。ただし論文の主張を「symbolic equation → neural weights」という一般論で立てず、**形式的意味保存を持つコンパイラの実装仕様と、数値・計算効率の定量的な追加価値**に絞るべきである。上のGate 0とGate 2が否定的なら、学習済みhypernetworkを必須部品にせず、検証済みsymbolic IR + 最適なsolverを選ぶハイブリッドcompilerへピボットする。
