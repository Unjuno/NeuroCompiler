# NeuroCompiler — Stage 7 実験・追試レポート

## 1. 結論

**実測事実**：微分を含む数式を正規化し、方程式残差だけで学習した生成器から、初期条件を満たす近似解ネットワークの重みを生成できた。未知の微分式に対する10更新後の解MSEではMetaがDirectPhysicsを上回った。

**同じく重要な否定的結果**：この一次元・線形ODEでは、最終層を残差の最小二乗問題として解く方が勾配更新より高精度だった。さらに、適切に調整した未学習ランダム基底とChebyshev法が学習済み基底を大幅に上回った。したがって「学習済み生成器が数値解法として必要／最良」という主張は支持されない。

**設計上の修正**：symbolic front-endの意味保存、neural back-endの近似性能、内側数値解法の性能、実装精度の四つを別々に評価する必要がある。

本パッケージは同一会話内で先に固定・実行されたStage 7の記録からコードを復元して再実行したもの。再実行を追加の独立10 seedとは数えていない。以下の値はこのパッケージ内に実在する再実行CSVから算出した。

## 2. 問題と変数表

本実験の座標・関数値・係数はすべて無次元。実際の物理単位を持つODE/PDEへの換算は行っていない。

|記号|意味|SI単位|定義|定義域・前提|型|
|---|---|---|---|---|---|
|\(x\)|独立変数|1|評価座標|\([-1,1]\)|スカラー|
|\(\lambda\)|ODE係数|1|解に掛かる係数|訓練0,0.5,1,2、通常試験0.25,0.75,1.5、外挿4,8|スカラー|
|\(b\)|初期値|1|\(u(-1)\)|生成範囲[-0.75,0.75]、実行時FP32定数|スカラー|
|\(e\)|入力数式|該当なし|Gaussian葉、加算、乗算、微分のAST|有限木|記号構造|
|\(f_e\)|ODE右辺|1|ASTを評価した関数|滑らかなGaussianの微分・和・積|スカラー値関数|
|\(u,u_{\rm ref}\)|厳密解・評価用参照解|1|初期値問題の解、数値積分による参照|参照はFP64、学習には使わない|スカラー値関数|
|\(D\)|微分演算子|1|\(d/dx\)|本実験は無次元座標|演算子|
|\(g_i,\mu_i,\sigma_i\)|Gaussian葉・中心・幅|1|下記定義|\(\sigma_i>0\)|関数・スカラー|
|\(\theta\)|生成NNパラメータ|1|1→32 tanh→1の97パラメータ|実数ベクトル、実装FP32|ベクトル|
|\(N_\theta,u_\theta\)|基本NN・初期条件付き解|1|下記定義|有限な重み|スカラー値関数|
|\(r_\theta\)|方程式残差|1|\(D u_\theta+\lambda u_\theta-f_e\)|連続関数、測定は点上|スカラー値関数|
|\(S\)|残差正規化係数|1|\(1+\operatorname{mean}(f_e^2)+(\lambda b)^2\)|正、右辺と初期条件のみから計算|スカラー|
|\(\mathcal R_k\)|更新後の正規化残差損失|1|query点での\(r^2/S\)平均|非負、inner更新k回後|スカラー|
|\(J\)|outer目的関数|1|(R5+βR0)/(1+β)|非負|スカラー|
|\(\beta\)|初期損失罰則係数|1|本実験0.25|非負|スカラー|
|\(m,s,t,k\)|手法・学習seed・task・更新回数|1|各添字|seed101–110、k=0,1,2,5,10,20,50,100|添字・整数|
|\(L_{m,s,t}(k)\)|解のMSE|1|参照解とのquery平均二乗誤差|非負、参照は評価専用|スカラー|
|\(N,N_T,N_S\)|query点数・group内task数・seed数|1|257、160、10|固定|整数|
|\(\epsilon\)|対数の数値安定化項|1|\(10^{-18}\)|正、主MSEに比べ十分小さい|スカラー|
|\(d_s,\Delta\)|seed効果・全seed平均効果|1|下記の対数MSE比|負が分子手法に有利|スカラー|
|\(\Delta_L,\Delta_U\)|95%区間端点|1|seed bootstrapの分位点|20,000再標本化|スカラー|
|\(u_A,k_{95}\)|seed成分の標準不確かさ・t係数|1|seed効果標準偏差/√10、t自由度9|条件付きType-Aのみ|スカラー|

\[
g_i(x)=\exp\!\left[-\frac{(x-\mu_i)^2}{2\sigma_i^2}\right],\qquad
u'(x)+\lambda u(x)=f_e(x),\qquad u(-1)=b.
\]

右辺の未見構造は積の微分、和と積を混ぜた微分、積の二階微分など。左辺は一貫して一次線形演算子であり、任意の微分方程式構造への一般化を試したわけではない。

**単位チェック**：x、幅、中心は無次元なので指数の引数は無次元。uとfも無次元、無次元座標に対する微分、λu、fの単位は一致する。MSE比の対数も無次元である。

## 3. 初期条件を満たす実行形式と証明

\[
u_\theta(x)=b+(x+1)N_\theta(x).
\]

x=-1を代入すると、
\[
u_\theta(-1)=b+((-1)+1)N_\theta(-1)=b+0=b.
\]
したがって有限なNN出力について、どの重みでも初期条件を満たす。生成97パラメータに加え、bを実行テンプレートの定数として保持する。97重みだけに初期条件情報まで埋め込んだという意味ではない。

FP32で格納したbに対して境界差は0。元の高精度bからFP32への丸めの最大絶対差は約2.98e-8であり、これは別の誤差源である。構成上境界条件を満たす試行関数は既存の設計原理である（Lagarisほか、physics/9705023）。

## 4. データ分割・訓練・計算条件

訓練2,048 task、validation192 task、開発120 task、最終480 task。最終はstructure／unseen_leaf／lambda_ood各160 taskで、4種類のASTを各40 task。学習seed101–110の10個を全条件で使用した。最終task bankはseed間で共通であり、4,800個の独立した問題を作ったのではない。

訓練Gaussianは中心{-0.7,-0.35,0,0.35,0.7}、幅{0.25,0.40}。未見葉は中心{-0.525,-0.175,0.175,0.525}、幅0.325。葉IDそのものはNNへ渡さず、中心・幅・中心二乗・微分階数などのdescriptorを使う。右辺energyは0.002以上になるよう生成時に制限する。

各taskは128 support点、訓練時には別の128 query点を持つ。最終評価queryは257点。生成器12,034パラメータ、生成対象97パラメータ。学習はCPU／FP32、Adam、outer LR0.001、task batch16、inner support32点、outer query64点。generator gradient normを1でclip。メタ勾配はfirst-order近似で、inner Jacobianを無視する。

|手法|outer更新数|inner更新数|target gradient評価数|目的|
|---|---:|---:|---:|---|
|DirectPhysics|3,600|0|3,600|現在の方程式残差を下げる|
|MetaPhysics|600|5|3,600|5回適応後の方程式残差を下げる|
|RegMeta|600|5|4,200|適応後残差と初期残差を両方下げる|
|SharedMeta|600|5|3,600|task descriptorを使わない共有初期値|

RegMetaは
\[
J=\frac{\mathcal R_5+\beta\mathcal R_0}{1+\beta},\qquad\beta=0.25
\]
をfirst-order近似で最適化する。Jは無次元のouter目的関数である。

参照解は積分因子と96点Gauss求積で生成する。正解解の値は訓練損失、checkpoint選択、学習率選択のいずれにも使わない。checkpointおよび学習率は既知分布のvalidation残差だけで選ぶ。学習率候補は0.0003/0.001/0.003/0.01/0.03。全手法の評価support系列は共通。

DirectとMetaで一致させたのはtarget gradient評価数であり、計算全体・forward回数・wall-clockではない。RegMetaはその数も多い。数値解法分岐は128 collocation点を一括使用し、10-step SGDは同じ128点プールから32点ずつ10回抽出する。異なるアルゴリズムなので同一計算量比較ではない。

## 5. 統計と判定

主評価はstructure群での10更新後の**絶対解MSE**。低初期損失を揃えることを必要としない実用性能比較である。
\[
L_{m,s,t}(k)=\frac1N\sum_{j=1}^N
\left[u_{\theta_{m,s,t}^{(k)}}(x_j)-u_{{\rm ref},t}(x_j)\right]^2,
\]
ここでjはquery点添字、\(\theta_{m,s,t}^{(k)}\)は手法・seed・taskごとのk更新後重み。
\[
d_s=\frac1{N_T}\sum_{t=1}^{N_T}
\log\frac{L_{{\rm Meta},s,t}(10)+\epsilon}
{L_{{\rm DirectBest},s,t}(10)+\epsilon},\qquad
\Delta=\frac1{N_S}\sum_s d_s.
\]
seedを単位として20,000回bootstrap、95%区間を分位点から計算。区間上端が0未満なら負の効果を支持、下端が0超なら逆方向を支持、それ以外はUNCERTAIN。非有意は「効果なしの証明」ではない。追試は同じ固定bankの対照・頑健性検査であり、独立した追加再現実験とは数えない。複数の副次比較の95%区間は同時信頼区間ではない。

### 主結果と初期損失罰則

| group       | contrast                    |   mean_log_ratio |   ci95_low |   ci95_high |   negative_seeds |
|:------------|:----------------------------|-----------------:|-----------:|------------:|-----------------:|
| structure   | Meta_vs_DirectBest_MSE10    |       -0.320181  |  -0.477912 |  -0.135578  |                8 |
| structure   | RegMeta_vs_DirectBest_MSE10 |       -0.389101  |  -0.594081 |  -0.162244  |                8 |
| structure   | RegMeta_vs_Meta_MSE0        |       -1.89337   |  -2.2928   |  -1.50521   |               10 |
| structure   | RegMeta_vs_Meta_MSE10       |       -0.0689201 |  -0.153681 |   0.0192632 |                7 |
| unseen_leaf | Meta_vs_DirectBest_MSE10    |       -0.454977  |  -0.627754 |  -0.265053  |                8 |
| unseen_leaf | RegMeta_vs_DirectBest_MSE10 |       -0.507098  |  -0.718011 |  -0.276424  |                8 |
| unseen_leaf | RegMeta_vs_Meta_MSE0        |       -2.11518   |  -2.53994  |  -1.69226   |               10 |
| unseen_leaf | RegMeta_vs_Meta_MSE10       |       -0.0521209 |  -0.130201 |   0.030865  |                7 |
| lambda_ood  | Meta_vs_DirectBest_MSE10    |        2.55651   |   1.16068  |   4.09119   |                2 |
| lambda_ood  | RegMeta_vs_DirectBest_MSE10 |       -0.423579  |  -1.35641  |   0.56748   |                6 |
| lambda_ood  | RegMeta_vs_Meta_MSE0        |       -4.32786   |  -4.59711  |  -4.06647   |               10 |
| lambda_ood  | RegMeta_vs_Meta_MSE10       |       -2.98009   |  -3.8836   |  -2.16658   |               10 |

負は行名の分子手法が有利。主比較は約-0.3202、幾何平均MSE比0.726で8/10 seedがMeta優位。RegMetaはMetaの初期MSEを改善するが、10-stepのRegMeta対Metaの区間は0を跨ぐ。したがって「初期性能を改善した」は支持されるが、「適応性能が完全に等しい」「必ずさらに速い」は主張しない。

### 絶対値（各seed内task中央値を取り、10 seed間で中央値・10–90パーセンタイル）

| method     |   step |   solution_mse_median |   solution_mse_p10 |   solution_mse_p90 |
|:-----------|-------:|----------------------:|-------------------:|-------------------:|
| DirectBest |      0 |           0.00515109  |        0.00371601  |        0.00600916  |
| DirectBest |     10 |           0.00257685  |        0.00169297  |        0.00295999  |
| DirectBest |    100 |           0.000729799 |        0.000618153 |        0.000800733 |
| Meta       |      0 |           0.0212319   |        0.0143207   |        0.0683055   |
| Meta       |     10 |           0.00159225  |        0.00143786  |        0.00181074  |
| Meta       |    100 |           0.000716363 |        0.00067512  |        0.000836508 |
| Random     |      0 |           0.0353512   |        0.0311146   |        0.0459229   |
| Random     |     10 |           0.00524747  |        0.00498786  |        0.00582002  |
| Random     |    100 |           0.00154025  |        0.00150798  |        0.00158322  |
| RegMeta    |      0 |           0.00365278  |        0.00308714  |        0.00574743  |
| RegMeta    |     10 |           0.00135553  |        0.00120627  |        0.00188327  |
| RegMeta    |    100 |           0.000733523 |        0.000663954 |        0.000805247 |
| SharedMeta |      0 |           0.0323062   |        0.0296538   |        0.0371087   |
| SharedMeta |     10 |           0.00384887  |        0.00369569  |        0.004546    |
| SharedMeta |    100 |           0.0014894   |        0.00146718  |        0.00155869  |

上の中央値比と、対数MSE比の幾何平均は異なる集約であり、同じ数字にはならない。

### loss matchingの不成立

MetaとDirect checkpointのvalidation初期残差を10%以内に揃えられたのは**0/10 seed**。従って「loss-matched initial stateの幾何を因果分離した」という解釈は今回無効。`DirectMatched`という保存ラベルは選択操作の名称にすぎず、matching成功を意味しない。さらに、残差の一致と解MSEの一致は別の条件である。

## 6. 発見から進めた追試：最終層の物理解法

学習済み隠れ層を固定すると、初期条件付き解の出力係数だけに関して方程式残差は線形になる。これをFP64のLAPACK gelsdで解き、validationでSVD cutoffを選んだ。FP32実行向けとFP64実行向けは別々に選択する。PyTorch 2.10の`torch.linalg.lstsq`仕様に従い、driverとrcondを明示した。

| method     | deployment   |   solution_mse |   rank |     max_abs_head |   naive_fp32_solution_mse |
|:-----------|:-------------|---------------:|-------:|-----------------:|--------------------------:|
| DirectBest | FP32aware    |    2.43144e-07 |     12 |   6430.63        |             nan           |
| DirectBest | FP64         |    9.16922e-09 |     14 | 191322           |               5.8458e-05  |
| Meta       | FP32aware    |    1.90539e-07 |     13 |   5072.32        |             nan           |
| Meta       | FP64         |    1.55326e-09 |     15 | 119032           |               3.96053e-05 |
| Random     | FP32aware    |    1.72644e-05 |     10 |  54858.8         |             nan           |
| Random     | FP64         |    3.00883e-07 |     12 |      1.40864e+06 |               0.00854219  |
| RegMeta    | FP32aware    |    3.25971e-07 |     13 |   7012.32        |             nan           |
| RegMeta    | FP64         |    3.34804e-09 |     15 | 143058           |               6.78748e-05 |

RegMetaの10-step SGDのMSE約1.36e-3に対し、128点の線形解法はFP32実行で約3.26e-7。しかし、ここで結論を止めると弱いbaselineに勝っただけになる。

### 未学習基底と古典解法を追加

未学習random tanh基底のスケールを0.7/1.5/3/6から、SVD cutoffとともにvalidation残差だけで選択した。隠れ層も出力重みも事前訓練しない。各問題について出力係数を物理残差から解く。Chebyshevは次数8/16/32からvalidation選択する。これらも正解解のサンプルを使わない。

| method                            |   solution_mse |   norm_residual |
|:----------------------------------|---------------:|----------------:|
| Chebyshev_FP32                    |    9.67686e-16 |     1.13521e-13 |
| Chebyshev_FP64                    |    1.67317e-16 |     1.11139e-13 |
| DirectBest_ForcingShuffled_Head32 |    3.26588e-07 |     1.19949e-05 |
| Meta_ForcingShuffled_Head32       |    1.76962e-07 |     4.46374e-06 |
| RandomTuned_Head32                |    1.83218e-12 |     2.37482e-11 |
| RegMeta_ForcingShuffled_Head32    |    2.94715e-07 |     7.65455e-06 |

調整random基底は約1.83e-12、Chebyshev FP32は約9.68e-16。機械精度近傍の値は入力丸め・参照精度に敏感なので、15桁の普遍的精度保証とは解釈しない。一方で、学習済み手法との差が非常に大きく、このベンチマークにおける優劣は丸め誤差だけでは逆転しない。

### 生成基底はtaskの意味を必要としているか

λとbを固定したまま、同じ群・λ内で右辺descriptorだけを置換し、正しい右辺で出力層を再度解いた。Direct/RegMetaの正しいdescriptorに統計的に明確な優位はなく、Metaでは逆方向も観測された。従って「学習済みのtask固有基底が不可欠」という説明は支持されない。これは全重みのzero-shot生成でdescriptorが不要という意味ではない。実際、Direct全重みを誤ったtaskから生成するとzero-shot解MSEは悪化する。

| contrast                                 |   mean_log_ratio |    ci95_low |   ci95_high |   negative_seeds | decision           |
|:-----------------------------------------|-----------------:|------------:|------------:|-----------------:|:-------------------|
| DirectBest_NaiveFP32_vs_FP64             |        8.89673   |   8.58506   |   9.20096   |                0 | POSITIVE_SUPPORTED |
| Meta_NaiveFP32_vs_FP64                   |       10.4384    |  10.0797    |  10.7899    |                0 | POSITIVE_SUPPORTED |
| RegMeta_Head128_FP32aware_vs_SGD10       |       -8.91613   |  -9.34399   |  -8.54153   |               10 | NEGATIVE_SUPPORTED |
| RegMeta_NaiveFP32_vs_FP64                |       10.1103    |   9.72155   |  10.433     |                0 | POSITIVE_SUPPORTED |
| Random_NaiveFP32_vs_FP64                 |        9.97307   |   9.30685   |  10.5567    |                0 | POSITIVE_SUPPORTED |
| DirectBest_HeadCorrect_vs_ForcingShuffle |       -0.037896  |  -0.146175  |   0.0663476 |                5 | UNCERTAIN          |
| Meta_HeadCorrect_vs_ForcingShuffle       |        0.117226  |   0.0597648 |   0.169705  |                2 | POSITIVE_SUPPORTED |
| RegMeta_HeadCorrect_vs_ForcingShuffle    |        0.0136337 |  -0.0428177 |   0.0677523 |                4 | UNCERTAIN          |
| RandomTuned_Head32_vs_RegMetaHead        |      -11.9077    | -13.442     | -10.2951    |               10 | NEGATIVE_SUPPORTED |
| RandomTuned_Head32_vs_MetaHead           |      -11.4052    | -13.0247    |  -9.70142   |               10 | NEGATIVE_SUPPORTED |

**推論**：この問題の主な制約は、記号知識を表す能力だけではなく、基底のスケール・条件数・内側解法にもある。MetaがSGDに有利な状態を作っても、最小二乗解法に最適な基底を作るとは限らない。

**既存研究との位置づけ**：parameterized differential equationの重み生成はHyperPINN（arXiv:2111.01008）、物理残差とランダム特徴の線形解法はPIELM（arXiv:1907.03507）に先行例がある。今回の追加対照を、それらの方式そのものの新発明として扱わない。

## 7. 数値精度による破れ方

RegMetaのFP64用解は出力係数が大きく、FP64実行ではMSE約3.35e-9でも、単純にFP32へcastすると約6.8e-5へ悪化する。対数比平均は約10.11で、幾何平均的には約2.46万倍の悪化。FP32実行を含めてcutoffをvalidation選択すると約3.26e-7まで改善する。

これは単純なNN capacity問題ではない。係数の相殺と丸めに依存するため、**重みと一緒にdtype・線形代数driver・cutoff・実行形式を管理する必要がある**。FP64でよい重みがFP32でもよい、とは限らない。

## 8. 係数外挿と発散対策

λ=4,8の外挿では、固定学習率が破綻した。307,200評価記録中8,313レコードでMSEまたは残差が非有限。すべてλ外挿群であった。これらを欠測として除外していない。

| method         |     rate |   rate_min |   rate_max |   n_fail |   n_cases |
|:---------------|---------:|-----------:|-----------:|---------:|----------:|
| DirectBest     | 0.268125 |    0.1625  |    0.33125 |      429 |      1600 |
| DirectFinal    | 0.2725   |    0.16875 |    0.3375  |      436 |      1600 |
| DirectMatched  | 0.3275   |    0.2625  |    0.39375 |      524 |      1600 |
| Meta           | 0.225    |    0.16875 |    0.2625  |      360 |      1600 |
| Random         | 0.325625 |    0.2625  |    0.375   |      521 |      1600 |
| RegMeta        | 0.204375 |    0.16875 |    0.28125 |      327 |      1600 |
| SharedMeta     | 0.1725   |    0.1375  |    0.20625 |      276 |      1600 |
| ShuffledDirect | 0.13875  |    0.0875  |    0.2     |      222 |      1600 |

分母は各手法160個のtask×10 seed=1,600ケースであり、1,600個の独立taskではない。

そこで、再学習せず、support残差がArmijo条件を満たすまでstepを半減する制御を追加した。最大LRは元のvalidation選択値のまま、1e-4の十分減少係数、最大20回半減。採択できない場合は元の重みを保持する。制御にquery誤差・正解解を使わない。

DirectBest／Meta／RegMetaについて、検査した全115,200評価記録で非有限値は0、元の重みを保持するfallbackも0、採択更新によるsupport損失の最大増加0だった。これは検査範囲での安定化であって、任意の方程式・dtypeの無条件安定性の証明ではない。support損失の単調性はquery損失や真の解誤差の単調性を保証しない。

## 9. H / T / D / C / U

|項目|内容|
|---|---|
|H|本設定の未知微分式で、MetaはDirectBestより10更新後の絶対解MSEが低い。別仮説として、学習済み特徴が強い非学習特徴より有利かを検査。|
|T|固定10 seed、構造／葉descriptor／λを分けた480-task bank。早期の有意差を理由とする停止なし。追試の同じbank再利用を明示。|
|D|主比較は95%区間上端が0未満でPASS。学習済み特徴の優位は強い対照で支持されず。loss-matching機構比較は検証不成立。|
|C|単純な線形・低次元問題なので、学習ではなく基底のスケールと古典数値解法で説明・解決できる可能性。|
|U|seedのばらつき、有限task bank、Gaussian family限定、FP32入力丸め、SVD rank選択、first-order近似、計算量の非一致。|

主比較のseed統計成分はu_A≈0.0934（無次元）。t分布の自由度9に対応するk_95≈2.262だが、本文のCIはこの積ではなくbootstrap分位点である。条件付きType-A成分だけを全誤差源の合成u_cと呼ばない。構造バイアス・分布選択まで含む総合u_cは未推定。

## 10. 実装検証と実行環境

| 検査                           |           結果 |
|:-------------------------------|---------------:|
| 残差勾配と自動微分の最大絶対差 |    8.88178e-16 |
| 残差式と出力微分の最大絶対差   |    8.88178e-16 |
| 表現された初期条件の差         |    0           |
| 参照積分とDOP853の最大絶対差   |    8.80241e-11 |
| Gauss64と96の最大絶対差        |    6.93889e-16 |
| IRと自動微分の最大絶対差       |    1.13687e-13 |
| 厳密rewrite一致組数            | 1200           |
| 非同値対照組数                 |  400           |
| 誤ったcollapse数               |    0           |

| 項目                        | 値                                                           |
|:----------------------------|:-------------------------------------------------------------|
| torch                       | 2.10.0+cpu                                                   |
| numpy                       | 2.3.5                                                        |
| pandas                      | 2.2.3                                                        |
| scipy                       | 1.17.0                                                       |
| matplotlib                  | 3.10.8                                                       |
| cpu                         | AMD EPYC 9V74                                                |
| visible_cpu_count           | 5                                                            |
| cpu_quota                   | 400000 100000                                                |
| clock_mhz_snapshot          | ['2596.134', '2596.134', '2596.134', '2596.134', '2596.134'] |
| clock_fixed                 | False                                                        |
| worker_threads              | 1                                                            |
| max_concurrent_workers      | 3                                                            |
| generator_parameters        | 12034                                                        |
| target_generated_parameters | 97                                                           |
| cuda                        | False                                                        |

単一seedあたり訓練時間は次の通り。最大3実験を並行、各worker1スレッド、クロック固定なし。学習ループ時間のみであり、データ生成・validation・評価・参照解・baseline調整の時間を含むend-to-end latency比較ではない。

| mode    |   seconds_median |   seconds_min |   seconds_max |   steps |   target_gradient_evals |
|:--------|-----------------:|--------------:|--------------:|--------:|------------------------:|
| direct  |          6.85606 |       6.71407 |       7.19401 |    3600 |                    3600 |
| meta    |          2.00086 |       1.89192 |       2.40593 |     600 |                    3600 |
| regmeta |          2.25762 |       2.1208  |       2.83877 |     600 |                    4200 |
| shared  |          1.62335 |       1.35456 |       1.95516 |     600 |                    3600 |

## 11. 研究上の判断

目標は知識から有用な実行可能neural stateを作ること。そのためには意味保存だけでなく、強い非学習解法に対する追加価値を示す必要がある。現在地は「限定された微分式と線形ODEで機能は動くが、精度面では学習の必要性が示せていない」。この線形例でseedだけを増やすより、従来法・調整random特徴・同等計算費用を伴う非線形または多次元の限定課題へ検証軸を移す方が情報価値が高い。ただし次の課題で勝つ保証はない。

|関連分野|転用可能性と条件|
|---|---|
|コンパイラ設計・形式手法|微分正規化IRと明示的な展開上限。意味保存と近似コード生成を分離する。|
|数値解析・Scientific ML|残差、条件数、SVD cutoff、実行dtypeを一つのsolver契約として管理する。|
|メタ学習・ハイパーネットワーク|初期性能・適応後絶対性能・内側数値解法との相互作用を別々の目的として評価する。|

## 12. 成果物と再現性

`results/confirmatory/seed101`～`seed110`に訓練済みtensor、task単位CSV、選択記録、訓練ログ、追試を保存。`analysis/`に再計算した統計と失敗数、`checks/`に単体検証、`figures/`に4図。解析は生CSVから再作成できる。`protocol.json`は復元した設定と再実行であることを記録し、外部事前登録ではない。

再実行の主比較平均対数比は約-0.320181042で、先の記録の6桁値-0.320181との差は約4.2e-8。非有限レコード8,313件も一致した。これを追加10 seedの証拠とは数えない。

## 13. ERROR CHECK

数学的保証は有理係数の指定rewrite規則と、表現されたbに対する初期条件に限定する。近似解の精度は実測であり証明ではない。matching失敗を成功として扱わない。非有限値を欠測扱いして性能をよく見せない。CIが0を跨ぐ比較を効果なしと断定しない。下限に近い数値精度を普遍的精度として主張しない。固定bankの追試・再実行を独立task数や独立seed数に加算しない。

## 14. 参考文献（一次資料）

1. Belbute-Peres, Chen, Sha. HyperPINN: Learning parameterized differential equations with physics-informed hypernetworks. arXiv:2111.01008 (2021).
2. Dwivedi, Srinivasan. Physics Informed Extreme Learning Machine (PIELM). arXiv:1907.03507 (2019).
3. Lagaris, Likas, Fotiadis. Artificial Neural Networks for Solving Ordinary and Partial Differential Equations. arXiv:physics/9705023; IEEE Transactions on Neural Networks 9(5), 1998.
4. PyTorch 2.10 official documentation, torch.linalg.lstsq: CPU gelsd driver、rcondによるeffective rank、明示的cutoffの推奨。

詳細な数学的導出は `proofs_ja.md`、実行手順は `README.md` を参照。
