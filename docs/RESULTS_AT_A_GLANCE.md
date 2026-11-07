# 主要結果一覧と統計上の注意

この表の値は**元アーカイブの集計報告・CSVに基づく**。保存モデルの学習を今回新規に走らせた結果ではない。95%区間は断りのない限り独立学習seed単位のbootstrapと元報告書が規定する統計量。Stage9Rの主比較は97.5%区間（Bonferroni補正）である。

| Stage | 対象と比較 | 元報告の主要効果量 | 95% CI等 | seed数 | 判断と限界 |
|---|---|---:|---:|---:|---|
| 0 | HyperInit / FO-MAML：0更新MSE | 0.460 / 0.916 | なし | 当初のseed表記に不整合 | zero-shot性能、Metaのfast adaptationの証明でない |
| 1b | h2 Meta − loss-matched Direct：relative log-AUC | +0.0015 | [−0.0251,+0.0309] | 20 | 優位未確認 |
| 1d | h10 Meta − loss-matched Direct：relative log-AUC | −0.7394 | [−1.1326,−0.4048] | 10 | 初期誤差matching条件で適応性を支持 |
| 1e | 未使用A+B+Cでh10 Meta−matched Direct | −0.6889 | [−0.9414,−0.4573] | 20 | OOD転移を支持。h20のmatching失敗を含まない |
| 2a | 8 primitive OOD：Meta−matched Direct | −0.3086 | [−0.3930,−0.2287] | 20 | 相対適応が速い。DirectFit-finalの絶対誤差優位は残る |
| 2b | 追加5構造平均：Meta−matched Direct | −0.5578 | [−0.7213,−0.4190] | 10 | 複数maskで相対適応を再現 |
| 3a | generic DeepSets：Meta−matched Direct | −0.0564 | [−0.2204,+0.1169] | 10 | FAIL/UNCERTAIN |
| 3c | 代数aware tokens：Meta−matched Direct | −0.4283 | [−0.5596,−0.3035] | 10 | 有利。Denseとの関数クラス等価性に注意 |
| 4d | Gaussian意味descriptor：Meta−matched Direct | −0.1539 | [−0.2215,−0.0946] | 10 | 同一Gaussian族の未見identityで支持 |
| 4d | Semantic / ID-only：zero-shot log-MSE比 | −1.9504 | [−2.0710,−1.8486] | 10 | 幾何MSE比約0.1422 |
| 5a | AST Tree / Bag：zero-shot log-MSE比 | −0.9544 | [−0.9719,−0.9393] | 10 | 幾何MSE比約0.385；Bagはoperator blind |
| 5b | 長期Meta−matched Direct | −0.0251 | [−0.0597,+0.0055] | 10 | 長期優位未確認 |
| 5d | fresh 1–10step early AUC差 | −0.0602 | [−0.0937,−0.0276] | 10 | 短期Meta優位のみ支持 |
| 6b | 同値rewrite間の最大生成weight差 | 0.0 | 構造的保証・5seed pilot | 5 | テストした可換半環の同値変換のみ |
| 7 | ODE 10 step Meta/Direct log-MSE比 | −0.3202 | [−0.4779,−0.1356] | 10 | initial-loss matching 0/10なので機構分離なし |
| 7 | tuned random feature / Chebyshev絶対MSE | 約1.83e−12 / 約9.68e−16 | 条件付き中央値 | - | 同タスク精度で従来法が圧倒。計算時間は異なる |
| 8a | 非線形ODE Meta/Direct log-MSE比 | +0.4200 | [+0.3720,+0.4679] | 10 | Metaはむしろ悪い |
| 8b | より長く学習したMeta/Direct | +0.0384 | [−0.0255,+0.1044] | 10 | Metaの優位を支持しない |
| 9R | 位置OOD・Weak/Data幾何相対L2比 | 0.9826 | 97.5% [0.9758,0.9896] | 20 | 同800更新では小改善、絶対誤差は依然大 |
| 9R | 位置OOD・Response/Data幾何比 | 0.9792 | 97.5% [0.9687,0.9890] | 20 | 同800更新で約2.1%改善 |
| 9R | 位置OOD・Response/DataTime幾何比 | 1.0471 | 95% [1.0310,1.0631] | 20 | 同学習時間の対照には不利 |
| 9R | 位置OOD：Response / Galerkin36相対L2中央値 | 101.66% / 3.05% | seed集計とtask集計で定義差 | 20 | 36係数でも直接数値法が大幅に高精度 |

## 指標の変数表

| 記号 | 意味 | SI単位 | 定義 | 条件 | 型 |
|---|---|---|---|---|---|
| \(L_m(k)\) | 手法mのstep kでのMSE | 1 | query上の平均二乗誤差 | 非負・無次元問題 | スカラー |
| \(k\) | 適応勾配更新回数 | 1 | checkpoint数 | 非負整数 | 整数 |
| \(K\) | 評価step集合 | 1 | stageが固定したcheckpoint | stageごとに異なる | 集合 |
| \(R_m\) | 初期状態に対する相対log-AUC | 1 | checkpointでのlog(MSE比)平均 | \(L_m(0)>0\) または安定化項 | スカラー |
| \(\Delta R\) | 2手法の相対log-AUC差 | 1 | \(R_{Meta}-R_{Direct}\) | 負ならMetaが相対的に速い | スカラー |

\[
R_m=\frac{1}{|K|}\sum_{k\in K}\log\frac{L_m(k)+\epsilon}{L_m(0)+\epsilon},\qquad \Delta R=R_{Meta}-R_{Direct}.
\]

\(\epsilon\) はlog計算の微小な無次元定数であり、stage実装の指定値に従う。**次元チェック**：MSEの比は無次元であるため対数とAUC差も無次元。stageごとに\(K\)や集約方法が異なるため、異なるstageの効果量を単純に縦比較してはならない。

## 公開前に再確認すべきこと

- Stage 1eの保存済み追試コードが不足。報告書と統計CSVは回収できるが、原コードの完全再現性は不明。
- Stage 3a/3bは一部aggregate CSVと過去の会話からの補足が中心。初期化交絡の議論を消さない。
- Stage 9旧版は原本なし。Stage 9Rは別実験。
- 指標の独立標本はtask数ではなく基本的に**学習seed**。同じ評価bankの繰り返しは独立実験ではない。
- 原報告書の結果を「独立再現された査読済み知見」と称さない。
