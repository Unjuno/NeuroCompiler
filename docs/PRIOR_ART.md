# 先行研究への接続（再検証が必要な参考文献表）

この文献表はこれまでの研究で扱った主要概念を整理したものです。**最新の文献網羅・優先権・新規性判定を保証するものではありません。** 論文投稿時はarXiv/DOI、著者、バージョン、公開日、実装条件を一次資料で再調査してください。

| 研究トピック | 例 | 本研究との関連 |
|---|---|---|
| Hypernetworks | Ha, Dai, Le, *HyperNetworks* (arXiv:1609.09106) — https://arxiv.org/abs/1609.09106 | 条件から別NNの重みを生成すること自体は既存 |
| Meta-learning | Finn, Abbeel, Levine, *Model-Agnostic Meta-Learning* (arXiv:1703.03400) — https://arxiv.org/abs/1703.03400 | 少数stepで適応しやすい初期状態の学習 |
| DeepONet | Lu et al., *Learning nonlinear operators via DeepONet* (arXiv:1910.03193) — https://arxiv.org/abs/1910.03193 | 関数から関数への作用素学習 |
| Fourier Neural Operator | Li et al., *Fourier Neural Operator* (arXiv:2010.08895) — https://arxiv.org/abs/2010.08895 | 離散解間に汎化するoperator model |
| HyperPINN | *HyperPINN* (arXiv:2111.01008) — https://arxiv.org/abs/2111.01008 | 物理方程式のパラメータ条件からPINN重みを生成 |
| PINO | Li et al., *Physics-Informed Neural Operator* (arXiv:2111.03794) — https://arxiv.org/abs/2111.03794 | 物理制約をoperator learningに組み込む |
| Deep Ritz | E, Yu, *The Deep Ritz Method* (arXiv:1710.00211) — https://arxiv.org/abs/1710.00211 | 変分汎関数・弱形式を利用する学習 |
| PDEformer / HyPINO / kanpiler | これまでの研究レビューに記載。正式な版・書誌情報は公開論文から要再確認。 | AST/方程式→モデル/重み生成という新規性の比較に不可欠 |

新規性を語る際は、次の命題を切り分けます。

1. 入力から重み生成（既存概念）。
2. AST/semantic IRと生成重みの対応（構文汎化と意味不変性は別）。
3. MetaInitの**loss-matched** short-horizon適応性（全てのstageで成立するわけではない）。
4. 強い数値法との同精度時間比較（未確立）。
5. 数値・記号的に検証できるソルバーの組合せ（研究構想）。
