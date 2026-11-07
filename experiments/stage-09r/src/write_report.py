from pathlib import Path
import json
import numpy as np
import pandas as pd
r=Path(__file__).resolve().parents[1]
s=pd.read_csv(r/'analysis/summary.csv'); c=pd.read_csv(r/'analysis/contrasts.csv'); t=pd.read_csv(r/'analysis/training.csv'); fresh=pd.read_csv(r/'analysis/fresh_contrasts.csv'); checks=json.loads((r/'checks/evaluation_audit.json').read_text())
labels={'id':'分布内','center':'未知の源位置','width':'狭い源','operator':'係数範囲外'}
sel=['Data','Weak','Response','DataTime','Galerkin36','Galerkin100']
tab=s[s.method.isin(sel)].pivot(index='method',columns='group',values='relative_l2_median').reindex(sel)[['id','center','width','operator']]*100; tab=tab.rename(columns=labels)
primary=c[c.primary][['numerator','denominator','ratio','ratio_low','ratio_high','wins','ci_level']]
time=t.groupby('method').agg(steps_median=('steps','median'),steps_min=('steps','min'),steps_max=('steps','max'),seconds_median=('train_seconds','median'),seconds_min=('train_seconds','min'),seconds_max=('train_seconds','max'))
floors=s[s.method=='Projection36'][['group','relative_l2_median']].set_index('group').reindex(['id','center','width','operator'])*100
floors=floors.rename(index=labels,columns={'relative_l2_median':'最小可能な相対L2誤差(%)'})
cor=s[s.method.isin(['Response','Response+Newton1','DataTime','DataTime+Newton1','Diagonal','Diagonal+Newton1','Diagonal+Newton2','Galerkin36'])].pivot(index='method',columns='group',values='relative_l2_median')[['id','center','width','operator']]*100
cor=cor.rename(columns=labels)
fsub=fresh[fresh.group=='center'][['numerator','denominator','ratio','ci_low','ci_high','wins']].copy(); fsub['ci_low']=np.exp(fsub['ci_low']); fsub['ci_high']=np.exp(fsub['ci_high'])
text=f'''# NeuroCompiler Stage 9R — 第一原理を使う学習の実測検証

## 結論

**同じ更新回数なら弱残差・接線応答を追加した学習は一部で改善した。しかし、同じ学習時間の教師学習を上回る実用的な一般化性能は示せなかった。** 未知の源位置での改善は幾何平均誤差比で約2%。依然として相対L2誤差の中央値は約100%である。

一方、誤差分解により、現在の大きな誤差の大部分は36基底そのものではなく係数生成にあると判明した。ただし、源の位置・幅のOODでは36基底だけで約3–4%の下限も残り、1%精度を狙うには基底を拡張する必要がある。

**物理法則を損失に追加すること、数値補正に使うこと、精度を検査することは別物。今回の差は、後二者を明確に分離しなければ理解できない。**

## 0. 来歴と実施範囲

これは今回実行した独立再実装Stage9Rである。過去に会話で報告されたStage9のコード・生データは、この環境のファイルと同梱ZIP内に確認できなかった。したがって過去のStage9数値を再現したとは主張しない。

主実験は20独立初期化/minibatch seed×4条件=80学習run。開発用seed100の4runは除外。教師問題1024、validation128、最終評価800（4群×200）、凍結済みモデルを使う別の400問題による追試を実行。二つの評価bankでseedを40に水増ししていない。

全コードとraw結果を同梱。`docs/protocol.md`と`checks/protocol_lock.json`は最終bank生成前の内部固定記録であり、外部の事前登録ではない。

## 1. 問題・学習条件

単位正方形、零Dirichlet境界の変係数・三次非線形楕円型PDE。右辺は正のGaussian源2個の和。式・変数表・導出は[proofs_ja.md](proofs_ja.md)。

|項目|固定条件|
|---|---|
|源の中心（train/ID）|各座標0.25–0.75|
|源の幅（train/ID）|0.10–0.18|
|源の振幅|10–50、正|
|反応係数の基準値|0–20|
|係数場の変動率|0–0.8|
|三次項係数|0–80|
|位置OOD|各座標0.06–0.20または0.80–0.94|
|幅OOD|0.04–0.07|
|係数OOD|反応係数25–50、三次項120–240|
|出力表現|各方向6個、計36個の直交正弦基底の係数|
|生成器|11→64 tanh→64 tanh→36、7268学習パラメータ|
|参照解|63×63内点、全正弦モードのNewton-CG、FP64|
|学習の物理求積|23×23内点|
|教師|参照解の36モード直交射影。Galerkin解と混同しない|
|学習|Adam、学習率0.001、batch64、800更新|
|モデル選択|独立validation上の係数誤差、100更新ごとと最終更新|

Dataは教師損失のみ。Weakは教師＋前処理付き弱残差。Responseはさらに一方の源振幅に対する接線応答残差を重み0.25で追加。全モデルの初期化とminibatch列はseed内で共通。

DataTimeは教師損失だけをResponseと同じ実測学習時間まで回す対照。更新回数は約3倍になる。時間合わせはFLOPs一致ではなく、import・bank生成・optimizer作成・保存を含まない学習ループとvalidationの比較である。

この実験はMetaInitの再検証ではない。4条件とも、問題descriptorから係数を直接予測する生成器の学習である。

## 2. 絶対精度

単位は相対L2誤差の%。NNは各seedのtask中央値を求め、その20seed間の中央値。数値法は同じ200taskの中央値。小さいほどよい。誤差100%は正解率0%という意味ではなく、誤差ノルムが解ノルムに匹敵することを意味する。

{tab.to_markdown(floatfmt='.4f')}

[図：絶対精度](figures/absolute_errors.png)

同じ36係数を使うGalerkin法との大きな差は、単に出力空間が狭いことだけでは説明できない。100係数法はさらに近似限界を下げる。ただし、この表は同じ実行時間での比較ではない。速度–精度Paretoの優位をここから結論してはいけない。

## 3. 主検定：改善は統計的にはあるが小さい

各seedで200個の位置OOD taskのlog相対L2誤差を平均し、条件差を取った。20seedをpaired bootstrap10000回で再標本化した。二つの主比較はBonferroni補正のため各97.5%区間。値は誤差比、1未満で分子が良い。

{primary.to_markdown(index=False,floatfmt='.6f')}

Weakは約1.7%、Responseは約2.1%の相対的改善。これは「誤差が数%まで下がった」ことではない。絶対誤差は約100%残っている。

時間を揃えると、位置OODのResponse/DataTime比は1.0471、探索的95%区間1.0310–1.0631。Responseの方が約4.7%不利。Response/Weak比較も、位置OODで接線損失の追加効果が明確とは言えない（詳細CSV参照）。

[図：paired比較](figures/paired_contrasts.png)

主検定の小さな差を、モデルが方程式の意味を理解した証拠、あるいは大幅な一般化改善と呼ばない。

## 4. 誤差分解

63×63離散参照解に対する直交射影を使い、総二乗誤差を「基底外成分」と「基底内係数生成誤差」に厳密分解した。直接格子計算との最大差は {checks['projection_identity_max_abs']:.3e}。

{floors.to_markdown(floatfmt='.4f')}

Responseの総二乗誤差に占める基底外成分の中央値は、位置OODで0.0935%、幅OODで0.3516%。逆に約99.91%、99.65%は係数生成側である。この割合は相対L2誤差ではなく、二乗誤差の構成比。

一方、36基底で達成可能な最良誤差自体が1%を超えるtaskは、位置OODで198/200、幅OODで200/200。生成器だけを改善しても、この下限は消えない。

**順序は、まず大きな生成誤差を減らし、その後に精度目標に応じて基底を増やす。両方を同じ原因として扱わない。**

[図：基底の限界](figures/basis_floor.png)

## 5. 追試：損失に入れるだけでなく実際に補正する

同じ36基底内でNewton方向を解き、エネルギーが減るように更新幅を制御した。補正に教師解は使わず、問題の係数と右辺のみを使用。求積は23×23。初期近似として、未学習の対角線形近似も追加した。

{cor.to_markdown(floatfmt='.4f')}

狭い源でResponseは64.39%から1回補正後4.18%へ改善。36基底の下限3.78%に近づく。位置OODでは101.66%から13.95%まで改善するが、まだ下限3.05%には達しない。

未学習の対角近似＋2回補正は、位置OODで3.11%、狭い源で3.78%。学習済み初期値が必要であるという証拠にはならなかった。ただし補正回数・計算量は異なり、NNと数値補正の総時間優劣は未測定。

この追試は主学習比較を見た後の探索解析。新しい汎用アルゴリズムを発明したとの主張ではない。

## 6. 別の400問題で固定モデルを再評価

すでに凍結した同じ80モデルを再学習せずに別bankへ適用。以下は位置OOD、探索的95%seed区間。

{fsub.to_markdown(index=False,floatfmt='.6f')}

小さい更新回数一致の改善と、時間一致での不利の両方が再現した。二つのbankしかないので任意のtask母集団への信頼区間とはみなさない。

## 7. 残差が小さいだけでは正確さを保証できない

Galerkin36の位置OODでは36モード内残差の中央値は約8.54e-15だが、解誤差は約3.05%。狭い源でも残差約1.65e-15に対して誤差約3.78%。これはバグではなく、試験空間外の成分を残差評価から除いているため。

全63×63モードで評価する負Sobolev残差から作った離散上界は、800taskで違反0件、誤差に対する最小比 {checks['min_discrete_bound_ratio']:.3f}。ただし参照解残差、求積・打切り、丸めを含む連続PDEの厳密証明ではない。区間演算による検証も未実施。

この点は「残差が小さければfallback不要」というrouterの設計に直接関係する。訓練に使う有限モードだけを検査すると、見えていない誤差を過小評価する。

## 8. 数値妥当性・再現性

- 直交性、誤差分解、エネルギー微分、Hessian、製造解、NNのJVP、物理JVP、入力検査：pytest9件に合格。
- 製造解での最大絶対誤差1.67e-16。これは指定した検査問題の結果であり、全taskの参照誤差ではない。
- 48taskを63→127格子で比較。最大相対差0.1172%、最大36係数射影差0.0946%。境界近くに源を置くと差が大きい。
- うち8taskを127→255格子でも比較。微小な連続精度を保証したとは言わない。
- 主結果64,000行、非有限値0。train/validation/testのparameter一致0件。
- Data・Response・DataTimeのseed0を再実行し、保存された全state tensorの差は0。DataTimeは記録した更新回数を固定して再実行した。
- 独立seedは20。800task×20を16,000個の独立学習モデルとして扱っていない。

## 9. 環境と計算量

AMD EPYC 9V74、可視5vCPU、CPU quota4コア相当。演算1thread、GPUなし。Python3.13.5、NumPy2.3.5、SciPy1.17.0、PyTorch2.10.0+cpu。生成器学習・出力FP32、係数からの関数評価と数値法FP64。クロック固定なし、学習中の周波数ログ未取得。決定論的torch演算とflush-denormalを使用。

{time.to_markdown(floatfmt='.6f')}

ここに含まれない費用：教師参照解生成、データbank作成、import、保存、推論用の右辺射影、最終数値補正、各種解析。訓練償却や総運用速度の主張には不十分。

## 10. H / T / D / C / U

- **H：** 弱形式・接線応答損失は、同じ36基底・同じ教師数で位置OODの係数生成を改善する。
- **T：** 20seed×4条件、最終800task＋独立400task、既知解・勾配・格子検証。
- **D：** 同更新数比較は小さい改善を支持。同学習時間における優位は支持せず、位置OODでは不利。接線損失の普遍的な追加効果も未確立。
- **C：** 改善は補助損失による正則化・計算量増加で説明できる可能性。参照ソルバーや基底の知識を追加することとNNの意味理解は別。
- **U：** 固定task bank、学習seed20、正のGaussian2源、固定領域・境界・作用素族・低次元入力に限定。損失重み、容量、長期学習、別作用素は網羅探索していない。合成不確かさu_c・包含係数kに基づく連続解精度保証は未算定。

## 11. 研究への判断

**事実：** 主な大誤差は係数生成にある。物理損失追加だけの効果は小さく、計算予算を揃えると優位ではない。数値補正は大きく改善し、有限基底の限界に近づく。

**推論：** 現段階では、すべてをNNへ学習させるより、既知の作用素を数値計算に利用し、生成器の追加価値を別に測る方が妥当。これは任意の機械学習solverに対する否定ではない。

**未検証：** 作用素に整合したアーキテクチャ、他の損失重み、より十分な学習、別の高次元PDEで同じ結論になるか。今回のモデルが「理解していない」という哲学的断定も行わない。

関連分野は、数値解析（射影・Galerkin・誤差推定）、機械学習（補助損失・OOD・計算予算対照）、コンパイラ/形式手法（実行形式・検証・fallbackの責務分離）。

## 12. 図・rawデータ

- [絶対誤差図](figures/absolute_errors.png)
- [主比較の区間](figures/paired_contrasts.png)
- [基底限界](figures/basis_floor.png)
- [具体的な解の断面](figures/solution_section.png)：seed0のDataTimeについて位置OODの中央値に最も近いtask370。選択規則を明示した説明用の例であり、代表性の証明ではない。
- [主統計](analysis/contrasts.csv)、[seed集計](analysis/seed_metrics.csv)、[全task](analysis/task_results.csv)
- [独立bank追試](analysis/fresh_contrasts.csv)、[残差検査](analysis/residual_indicators.csv)
- [数式と証明](proofs_ja.md)、[再現手順](README.md)

## 方法上の参考

数値基底の実装はSciPyの直交DST-I仕様に従い、Gram行列でも独立検査した。物理・変分情報を学習へ組み込む方向にはDeep Ritz等の先行研究がある。この実験の補助損失やNewton補正自体の新規性を主張しない。

- https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dst.html
- E & Yu, The Deep Ritz method, arXiv:1710.00211, https://arxiv.org/abs/1710.00211
'''
(r/'report_ja.md').write_text(text)
print('report chars',len(text))
