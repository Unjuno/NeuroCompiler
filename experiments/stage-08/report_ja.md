# NeuroCompiler Stage 8 — 非線形ODE・強い数値対照・solver routing の実験報告

## 0. 研究上の結論

**実測**：`u' + λu + βu³ = f` の無次元一次非線形IVPに対して、記号式から生成したNN重みのzero-shot推論・残差のみを使った適応は実装できた。しかし、Meta学習済み初期値がDirectFitより良いとする仮説は、最初の10-seed試験では**逆方向**だった。meta学習予算を増やして完全別のtask bankで追試すると差は縮小したが、Meta優位は支持されなかった。

**新しい発見**：強い非学習法を追加すると、Chebyshev-Lobatto配置の非線形最小二乗法が約`10^-18`の解MSE中央値を達成した。さらに、**定数初期値＋数値的Newton補正3回**は、完全別bankで`MSE<10^-5`を**240/240 task**で満たし、NN重み＋1回補正（78.5–82.1%）より高速だった。したがって、この限定ODE族では学習済みweight compilerの*実用上の必要性は立証されない*。

結論は **「NeuroCompilerの数学的意味保存front-endを維持し、バックエンドをsolver特性で選ぶ」** 方向へ更新すること。汎用非線形ODE／PDEや未知演算子への一般化を主張しない。

## 1. H / T / D / C / U

|項目|内容|
|---|---|
|H|Stage8a：未知の非線形微分式においてMeta300がDirect1200より10更新後の絶対解MSEを改善する。Stage8b：開発で増やしたMeta900が再び優位になる。Stage8d：固定された非学習補正器が新規taskに対して所定精度で有効か。|
|T|Stage8aと8bは**別々の未使用test bank**（各240 task、3群×80）。訓練seed301–310の10独立モデル。Stage8dはさらに別の240-task bank、学習済み重みのみ再利用。|
|D|8aのMeta優位は**FAIL（逆方向）**。8bは**UNCERTAIN（CIが0を跨ぐ）**。8dの定数＋3回補正は**固定bank上の評価基準ではPASS（240/240）**だが、一般的な保証はない。|
|C|非線形化だけでは学習が必要にならない。計算予算／内側学習率、数値解法、点配置が支配因子である可能性。|
|U|seed間変動、task分布、ベースラインのfloat精度・collocation点、時間計測のバッチ依存性、数値積分誤差。後半の追試は探索的、全ての副次比較に同時信頼区間は付けていない。|

## 2. 問題定義・記号・SI単位

この実験は完全に無次元化している。`x`, `u`, `f`, `λ`, `β`, `b` は全てSI単位1。

|記号|意味（日本語）|SI単位|定義|定義域／前提|型|
|---|---|---|---|---|---|
|`x`|独立変数|1|領域座標|[-1,1]|スカラー|
|`u(x)`|真の解|1|一意のIVP解|滑らかな実関数|関数|
|`f(x)`|記号式から得る右辺|1|Gaussian leafと積・和・微分のAST|本実験の構文集合|関数|
|`λ`|線形項係数|1|与えられたODE係数|非負、訓練0–2・OOD3/5|スカラー|
|`β`|非線形項係数|1|`u³`の係数|正、訓練0.25–1.5・OOD2/3|スカラー|
|`b`|初期値|1|`u(-1)`|[-0.6,0.6]|スカラー|
|`θ`|生成重み|1|tanh NN 1→32→1、97個|有限な実数（通常FP32）|ベクトル|
|`Nθ(x)`|基本ネットワーク|1|重みθで指定|有限出力|関数|
|`uθ(x)`|初期条件を満たす近似|1|`b+(x+1)Nθ(x)`|x∈[-1,1]|関数|
|`r(x)`|物理残差|1|`uθ'+λuθ+βuθ³-f`|可積分|関数|
|`δ(x)`|真の解と近似の差|1|`u−uθ`|δ(-1)=0|関数|
|`a(x)`|Newton線形化係数|1|`λ+3βuθ²`|非負|関数|
|`k`|適応step数|1|gradient update回数|0,1,2,5,10,20,50,100|整数|
|`L`|解の二乗誤差|1|129点上の平均二乗誤差|非負|スカラー|
|`Δ`|seed間平均対数MSE比|1|Meta/Directの対数比をtask→seed順に集約|実数、負ならMeta優位|スカラー|
|`T`|実時間|s|CPU wall-clock|非負|スカラー|

検証した方程式：

\[
u'(x)+\lambda u(x)+\beta u(x)^3=f(x),\qquad u(-1)=b.
\]

NNは \(u_\theta(x)=b+(x+1)N_\theta(x)\) とし、重みに依存せず初期条件を正確に満たす（FP32格納された境界値について）。訓練・チェックポイント選択・LR選択には**参照解を用いない**。参照解はDOP853 FP64による評価専用。**次元チェック**：無次元座標の微分項・線形項・三次項・右辺はいずれも単位1、MSEと対数比も無次元。

## 3. Stage8a — 事前固定の10 seed試験

- train1152 task、validation96 task、development72 task、final240 task（3群×80）。
- train kinds：Gaussian、sum、product、Dg、Dsum、D²g。final kinds：Dproduct、Dmix、D²product、Dproduct_plus。
- Stage8a test乱数seed823005（先行の試験seed823004はpilotとして隔離）。
- Direct：outer1200回。Meta：outer300回、inner horizon3、inner LR0.003。RegMetaでは初期残差罰則0.25。各stepで16 taskを抽出。
- 評価では方法間で同一のsupport minibatch系列、各taskで勾配norm上限30、最大10回半減のsupport-only backtracking。選択したLRはvalidation上で評価、テストには使用しない。
- **主指標**：structure群での10更新後のseed別task平均log(解MSE Meta / DirectBest)、最終10 seedの平均。CIは**seed cluster**を2万回bootstrap。

主結果は \(\Delta=+0.4200\)、95% CI `[+0.3720,+0.4679]`、**10/10 seedでMeta不利**。幾何平均MSE比は `exp(0.4200)=1.522`。RegMetaも同じ方向。NN gradient式の自動微分最大誤差 `7.11×10^-15`（FP64）。Stage8aはMeta優位仮説のFAIL。

## 4. Stage8b — 失敗原因を開発bankで探索し、別bankで追試

Stage8aの結果を見た後、**最終bankには使わず**、developmentでMeta horizon=1/3/5、inner LR=0.003/0.01/0.03、学習budget300/900を比較。Meta900、horizon3、LR0.003を開発段階で選択した。適応評価LR候補は0.001–0.8へ拡張。**Stage8b最終bankは新規seed823006で生成**。Direct1200はそのまま。Meta900は1 task当たり3600 target-gradient評価で、Direct1200の3倍（optimizer wall-clockは近い）。

### 主結果

|学習方法|10-step相対結果（structure OOD）|95% seed CI|Meta優位seed|解釈|
|---|---:|---:|---:|---|
|Stage8a Meta300 / Direct1200|+0.4200|[+0.3720,+0.4679]|0/10|Metaは明確に悪い|
|**Stage8b Meta900 / Direct1200**|**+0.03835**|**[−0.02555,+0.10442]**|**2/10**|統計的に優位を示せず|

Stage8bのstructure群（10 seed中央値の、seed内task中央値）：

|モデル|0-step MSE|10-step MSE|100-step MSE|
|---|---:|---:|---:|
|DirectBest|0.00287|0.00132|0.000631|
|Meta900|0.00345|0.00134|0.000609|
|RegMeta900|0.00341|0.00134|0.000610|
|Random|0.0454|0.00208|0.00158|

**注**：10-stepの中央値比較とtask log-ratioの差は異なる統計量。初期loss matchingは実施していないため、最適化geometryを因果的に分離した結果とは扱わない。100-step絶対MSEが近いことは、同等性の証明ではない。

### 計算条件・時間

今回の実行環境：Intel Xeon Platinum 8573C、約2.300GHz（表示値、固定clockではない）、CPU quota4core相当、CPU affinity5 vCPU、各workerでPyTorch/BLASは1thread、Linux 6.18、Python 3.13.5、PyTorch 2.10.0+cpu、SciPy 1.17.0、NumPy 2.3.5。

学習時間の10 seed中央値（範囲）：Direct1200が**2.216秒**、Meta900が**2.529秒**、RegMeta900が**2.831秒**。訓練batchは16 task、NNはFP32。これらはouter training loopの時間のみで、task bank構築やvalidation・final評価は含めない。

## 5. 強い非学習ベースライン（数値法）

**同じ128 support点**でChebyshev多項式出力を用意し、FP64の非線形最小二乗（Levenberg–Marquardt）で残差を最小化。degree候補8/12/20/32はdevelopment残差中央値だけから選択（結果degree32）。Stage8b 240 taskでは解MSE中央値が**7.82×10^-15**。ただし最大誤差は6.13×10^-5で、**ランダムsupportの不規則配置が原因候補**だった。

### 2×2要因実験

同じ次数32、同じsolverで、support位置と右辺計算精度だけを切り替えた（240 task、**探索的事後解析**）。

|support点位置|f入力型|中央値MSE|最大MSE|MSE>10^-7のtask数|
|---|---|---:|---:|---:|
|ランダム128点|FP32|6.58×10^-15|**6.12×10^-5**|3/240|
|ランダム128点|FP64|6.61×10^-15|**6.13×10^-5**|3/240|
|Chebyshev-Lobatto 128点|FP32|4.49×10^-18|6.41×10^-13|0/240|
|Chebyshev-Lobatto 128点|FP64|3.41×10^-18|6.41×10^-13|0/240|

したがって今回の誤差tailの主因は**FP32丸めではなくcollocation点配置**。Stage8dの別の240 taskでもCheb-Lobatto解MSEは中央値**1.53×10^-18**、最大**1.94×10^-13**と再現した。FP32出力に量子化しても全240 taskでMSE<10^-5を達成した。

これは平滑で単調散逸的な、今回の一次元ODE族に関する結果である。高次元・特異摂動・不連続・複雑境界のPDEには一般化できない。

## 6. Stage8c/8d — 数値的欠陥補正（defect correction）

### 厳密な代数

近似解を \(v\)、残差を \(r=v'+\lambda v+\beta v^3-f\)、真の解との差を \(\delta=u-v\) と置く。真の方程式から残差式を引き、三次式を展開すると、

\[
\delta' + (\lambda+3\beta v^2)\delta = -r-\beta(3v\delta^2+\delta^3),\qquad\delta(-1)=0.
\]

二次以上の項を捨てて線形誤差方程式を解くのがNewton型欠陥補正である。この線形ODEは積分因子で解ける。実装は129等間隔点で指数中点更新し、3回まで補正する。**数値積分の離散化誤差は残る**ので、厳密なerror certificateではない。

### 新しい未使用task bankでの結果（Stage8d、seed823007）

Stage8cの初回探索の後、**補正回数を固定してから**別の240-task bankを生成。Stage8bで学習した10 seedの重みを再利用した。MSE<10^-5の達成率を評価：

|方法|最終bankでの達成率|解MSE中央値|
|---|---:|---:|
|**定数 `u=b`＋2回補正**|236/240 = **98.3%**|2.11×10^-9|
|**定数 `u=b`＋3回補正**|**240/240 = 100%**|**6.37×10^-10**|
|Direct学習済み重み＋1回補正|82.1%（10 seed×240固定task）|5.74×10^-7|
|Meta学習済み重み＋1回補正|78.5%（同上）|7.76×10^-7|
|Chebyshev-Lobatto 32次|**240/240 =100%**|1.53×10^-18|

**定数初期値は学習コスト0**。したがって「NNは定数法よりNewton補正を1回節約するから高速」という仮説は、今回は不成立だった。

### CPU throughput（同じ事前計算済みbank、warmup5回、反復30回、1スレッド）

|方式|240-taskバッチの中央値|1task換算|精度／備考|
|---|---:|---:|---|
|NN zero-shot（生成＋129点評価）|14.54 ms|60.6 µs|近似精度は低い|
|NN＋1回補正|21.97 ms|91.6 µs|達成率82%（Direct）|
|**定数＋2回補正**|**7.88 ms**|**32.8 µs**|達成率98.3%|
|**定数＋3回補正**|**10.85 ms**|**45.2 µs**|達成率100%|
|Cheb-Lobatto 32次（逐次処理）|約0.77 s/240 task換算|約3,200 µs|高精度、逐次処理計測|

batch比較はベクトル化の恩恵を受ける。ChebはPythonで逐次解いており、solver実装・配列化・メモリ転送・前処理時間が完全に公平な設定ではない。上記時間は**symbolic AST正規化、task bank準備、disk I/Oを含まない**。CPU clockは固定していない。NNはFP32生成重みをFP64へcastして補正した。

### 安定な散逸ODEが非学習法に有利な理由（証明）

近似誤差を \(e=v-u\) と置くと、

\[
e'+q(x)e=r,\quad q=\lambda+\beta(v^2+vu+u^2),\quad e(-1)=0.
\]

ここで \(v^2+vu+u^2=(u+v/2)^2+3v^2/4\ge0\)。 \(\lambda,\beta\ge0\) なので \(q\ge0\)。積分因子を掛け、初期条件を適用すると、

\[
e(x)=\int_{-1}^{x}\exp\!\left(-\int_t^x q(s)ds\right)r(t)dt.
\]

指数因子は0以上1以下。ゆえに三角不等式・Cauchy–Schwarzから

\[
|e(x)|\le\int_{-1}^x|r(t)|dt\le\sqrt{2}\biggl(\int_{-1}^1r(t)^2dt\biggr)^{1/2}.
\]

両辺を二乗し区間長2で平均すると、真の連続平均二乗誤差は連続残差平均二乗値の**4倍以下**。ただしこの導出は連続積分が正確な場合で、129点の離散残差値を代入しただけでは**厳密な数値証明にはならない**。

この非負減衰性・滑らかさ・区間長2は本実験の有利な構造であり、実際の不安定PDEや強い剛性を持つ系は未検証。

## 7. ERROR CHECK（再現・境界・統計）

- 手計算残差勾配 vs FP64 autograd：最大`7.11×10^-15`。
- 異なる数値積分法Radauとの参照照合：12 task上、最大絶対差`1.67×10^-10`。相対差・評価分解能は別評価が必要。
- train/test IR hashの完全一致は0件。Stage8a/8b/8dのtest bank乱数seedは823005/823006/823007で分離。
- 10独立**学習seed**。240 test taskは各seedで共通なので、2400件を独立標本としてbootstrapしていない。
- Stage8a/8bでは非有限MSE 0件、support-only backtrackingを使い、失敗したgradient stepは**重み保持**として数えた。学習時のinner勾配はnorm30でclip。
- Stage8bは初期lossをDirectとmatchingしていないので、optimization-friendly basinの機構仮説には使わない。
- Stage8cのNewton補正は事後発見。Stage8dは補正数を事前固定した別bank追試だが、依然同じ合成ODE generatorに従っている。
- Stage8dのNN評価は固定240task×10独立学習seed、定数法/古典法はseed非依存。非学習baselineのtaskを10回繰り返して独立性を水増ししていない。
- `solve_ivp`参照解、実測誤差閾値、FP32/FP64動作、clock非固定、batch条件はレポート中に明示。

## 8. 研究ロードマップへの更新

1. **P0 完了**：この単調散逸的な一次元非線形ODEでは、学習済みweight compilerの精度・時間優位は確認できない。classic numerical backendの方が優位。
2. **P1 次段階**：非学習法が弱い可能性のある**空間次元2以上・可変領域・多数の非線形作用素族**へ対象を変える。ただし各領域・精度条件に対しFEM/FVM/FFT/スペクトル法など最適な非学習baselineを用いる。
3. **P2**：新しい対象では精度別のPareto曲線（0-shot、1–10 step、定数解＋補正、古典法、学習済み生成器）を主指標とし、単一MSE中央値の比較を禁止。
4. **P3**：記号正規化器で意味保存を引き続き検証する。solver選択をfront-endへ組み込み、学習による価値が確認された場合のみNN backendを選ぶ。
5. **P4**：数値誤差を扱う正式な実装契約（dtype、backend、ABI、求積、補正停止条件、精度保証／保証不能の区別）を追加する。

### 今回作成した動作プロトタイプ

`solver_router.py` に、記号的canonical polynomialの入力に対して `mode="fast"`（定数＋3回Newton補正）と `mode="accurate"`（Cheb-Lobatto32＋非線形最小二乗）を選択するルーターを実装した。**名前は「モード」であって誤差の保証ではない**。この小さいODE族を扱う開発用prototypeであり、自動PDE solverではない。

## 9. 参考文献との関係

Stage7で強い非学習基底を追加した動機はPIELM (arXiv:1907.03507) などの既存Physics-Informed Extreme Learning Machine研究と整合する。パラメータ付き微分方程式向けhypernetworkはHyperPINN (arXiv:2111.01008) に先行例がある。したがって重み生成一般を新発明として扱わない。今回独自に試験したのは、**この限定benchでの学習必要性の反証、数値point-placement交絡、非学習Newton補正を含む経済性比較**である。
