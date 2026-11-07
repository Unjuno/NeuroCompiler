# Knowledge-to-Weights Compiler 研究引き継ぎ書

## 0. 一文で何をする研究か

**関数・方程式・規則・既存知識を入力すると、その知識を学習しやすいニューラルネットワークの初期パラメータへ変換する「変換器」を学習し、未知の知識構造に対してもゼロから学習するより速く適応できるかを検証する研究。**

---

## 1. 研究の中心目的

現在のニューラルネットワークは、多くの場合、大量データと勾配降下によって既知の構造や法則まで再発見する。

本研究では、

```text
既存知識 K
  ↓
Knowledge / Function-to-Weights Generator G
  ↓
対象NNの初期パラメータ θ0(K)
  ↓
少量の追加学習
  ↓
最終パラメータ θ*(K)
```

という経路を学習する。

狙いは「完成したNNを一発生成すること」ではなく、

**未知タスクに対して、最適解に近い、または最適化しやすい初期状態を生成すること。**

これにより、既知の知識をニューラルネットワークへ「事前投入」し、毎回まっさらな状態から再学習するコストを下げる。

---

## 2. 研究の重要な区別

### 2.1 既存研究に近い部分

- Hypernetwork: NN/条件から別NNの重みを生成
- Meta-learning: 多数のタスクから良い初期値を学ぶ
- Neural Operator: 関数から関数への写像を学ぶ
- LoRA/Adapter generation: 条件やタスク記述からパラメータ差分を生成

したがって、「入力からNN重みを生成する」だけでは新規性は弱い。

### 2.2 本研究で狙う新規性

新規性の中心は次の組合せに置く。

1. **構造化された知識・関数・法則を入力する**
2. **その知識を利用しやすいNN初期状態へ変換する**
3. **学習時に見ていない知識の組合せへ compositional generalization する**
4. **完成重みではなく、少量学習で完成する「良い初期状態」を生成する**
5. 将来的には **形式知識 → LLM LoRA/Adapter/初期状態** へ拡張する

研究の強い形は、

> **Compositional Formal-Knowledge-to-Model Compiler**

と表現できる。

---

## 3. 何を入力し、何を出力するか

### 入力候補

- 解析関数
- ODE/PDE
- 境界条件
- 保存則
- 数学的制約
- 文法規則
- 言語類型情報
- 因果関係
- その他の構造化知識

### 出力候補

最初は小型NNの全パラメータ。

将来的には大規模モデルの、

- LoRA
- Adapter
- Parameter delta
- 一部の層
- 初期状態

を出力する方が現実的。

---

## 4. 中心仮説

### H

未見の構造的組合せに対しても、問題条件から生成した初期値 θ0(K) は、Random initialization や共有初期値より少ない追加学習で同等精度に到達する。

### 最も重要な点

単なる「未知係数への補間」では不十分。

例えば primitive を A, B, C とした場合、

```text
Train:
A
B
C
A+B
B+C

Test:
A+C
A+B+C
```

のように、**構造の組合せ自体を hold-out** する。

これに成功することが、本研究の主張を強くする。

---

## 5. なぜ解析可能な問題から始めるか

解析解を持つ問題なら、正解データを大量に自動生成できる。

```text
問題仕様 K
  ↓
解析解 f_K
```

このため、

- 教師データの生成コストが低い
- ground truth が明確
- 未見構造を人為的に設計できる
- train/test leakage を厳密に管理できる
- モデル能力と汎化能力を切り分けやすい

という利点がある。

重要なのは、「正解NNの重み」を教師にする必要がないこと。

同じ関数を表すNN重みは非一意なので、

```text
生成重み ≈ ある正解重み
```

ではなく、

```text
生成NNを実行
  ↓
解析解との誤差
  ↓
必要なら数step適応
  ↓
適応後の誤差
```

を損失にする。

---

## 6. Stage-0 実験

### 6.1 解析関数

3つの primitive を使用。

```text
A = sin(pi x)
B = x
C = 2x^2 - 2/3
```

対象関数は、

f_c(x) = c1 A(x) + c2 B(x) + c3 C(x)

active primitive の係数は符号付きでおよそ 0.5〜1.5。

### 6.2 Train/Test 構造

Train:

```text
A
B
C
A+B
B+C
```

Test:

```text
A+C
A+B+C
```

Test の active-component pattern は meta-training 中に未出。

### 6.3 Target NN

```text
x
↓
Linear(1,16)
↓
tanh
↓
Linear(16,1)
```

49 parameters。

### 6.4 Weight Generator

```text
(c1,c2,c3)
↓
Linear(3,32)
↓
tanh
↓
Linear(32,49)
+
shared base parameter
↓
θ0(c)
```

### 6.5 比較

- HyperInit: 問題依存の初期値生成
- FO-MAML_shared: 全問題共通の meta-learned initialization
- Random: ランダム初期化

### 6.6 実行条件

- Python 3.13.5
- PyTorch 2.10.0+cpu
- CPUのみ
- seed = 3
- meta steps = 300
- meta batch = 4
- meta inner steps = 2
- adaptation learning rate = 0.05
- support points = 12
- test samples = 480 / method / adaptation-step aggregation
- first-order meta-gradient approximation

---

## 7. Stage-0 結果

未見構造 A+C および A+B+C をまとめた median MSE:

| Adapt steps | HyperInit | FO-MAML | Random |
|---:|---:|---:|---:|
| 0 | 0.460 | 0.916 | 0.939 |
| 1 | 0.458 | 0.874 | 0.918 |
| 3 | 0.450 | 0.838 | 0.887 |
| 5 | 0.448 | 0.787 | 0.870 |
| 10 | 0.439 | 0.692 | 0.807 |

### 解釈

弱い意味では成功。

- 未見 composition に対し、HyperInit は全 step で最良
- 0-step でも共有初期値・random よりかなり低い MSE
- 問題依存 initialization に意味があるシグナルは得られた

ただし、強い意味ではまだ未達。

設定した MSE <= 0.10 の到達率は、

- HyperInit: 約1.5〜1.7%
- FO-MAML: 0%
- Random: 0%

したがって、

**「未知構造を高精度に一発変換できた」とは言えない。**

現時点の判定:

> PROMISING / NOT PROVEN

---

## 8. 現時点で排除できていない対立仮説

1. Generator が係数から関数を直接近似しているだけ
2. 「良い optimization basin」を生成しているわけではない
3. primitive 数が3しかなく簡単すぎる
4. test composition が2種類しかない
5. FO-MAML baseline の最適化不足
6. target architecture に強く依存している
7. 未知構造ではなく、既知 primitive 間の単純補間に近い
8. weight-space symmetry の扱いが未整理

---

## 9. 次に行うべき実験

### Stage-1: 「良い初期値」仮説の分離

最優先。

HyperInit が単に 0-step で良い関数を出しているだけなのか、それとも **その後の学習が速い初期状態** を生成しているのかを分離する。

#### 推奨設計

- HyperInit と baseline の 0-step loss を意図的に近づける
- その状態から 1, 5, 10, 50, 100 step の収束曲線を比較
- 同一初期lossからどちらが速く最終精度へ到達するかを見る

これが通れば「initialization compiler」という主張が強くなる。

---

### Stage-2: primitive 数を増やす

3 primitive から、

- 8
- 16
- 32

へ増やす。

全組合せを学習させず、組合せを大規模に hold-out。

評価:

- seen primitive / unseen composition
- higher-order composition
- extrapolation
- zero-shot
- few-step adaptation

---

### Stage-3: 入力表現を簡単な係数ベクトルから構造表現へ変更

現在は c=(c1,c2,c3) が強すぎる入力表現。

次は、

- token列
- expression tree
- symbolic AST
- operator graph
- equation tokens

などを入力にする。

Generator は Transformer / graph encoder を候補とする。

---

### Stage-4: ODE

解析可能な ODE family に移行。

例:

- 線形一次
- 線形二次
- 強制振動
- damping
- coupled systems

係数だけでなく、演算子構造を hold-out。

---

### Stage-5: PDE / Scientific ML

- diffusion
- advection
- reaction
- wave
- boundary-condition composition

を primitive として扱い、未知組合せを test。

---

### Stage-6: 形式知識 → LLM parameter delta

Scientific toy domain で成立した後、

```text
言語学的知識
↓
Knowledge Encoder
↓
LoRA Generator
↓
base LLM
↓
少量の corpus
```

へ拡張。

LLM 全重み生成から始めず、LoRA / Adapter を推奨。

---

## 10. 評価指標

主指標:

1. 同一 test error 到達までの gradient steps
2. sample efficiency
3. FLOPs
4. wall-clock
5. zero-shot error
6. few-shot / few-step error
7. unseen-composition performance

副指標:

- seed variance
- generator parameter count
- adaptation stability
- target architecture transfer
- coefficient extrapolation
- primitive-count scaling

---

## 11. PASS / FAIL / UNCERTAIN

### PASS

未見 composition において、同一初期lossまたは公平な条件から、提案法が baseline より少ない step/sample/FLOPs で所定精度に到達。

### FAIL

既知 task では改善するが、未見 composition で advantage が消える。

### UNCERTAIN

seed variance、baseline tuning、計算量差などで結論が分離できない場合。

---

## 12. 研究上の重要な原則

- 「重みを生成できた」だけでは研究主張にしない
- 未見 composition を test の中心にする
- 完成重みより「学習しやすい初期値」を狙う
- weight-space MSE を主要lossにしない
- behavior-space / post-adaptation loss を使う
- baseline を十分調整する
- 最初からLLMへ行かず、小さい解析系で反証可能性を高める
- 「未知」を係数・外挿・構造・未知primitiveに分けて報告する
- 研究の価値は「既存知識をパラメータとして投入し、再学習を減らせるか」にある

---

## 13. 最終的な研究ビジョン

最終形は、

```text
人間が持つ既存知識
    ↓
汎用 Knowledge Compiler
    ↓
実行可能な Neural Representation
    ↓
少量の経験・データだけで適応
```

という学習パラダイム。

狙っているのは、

**「データからすべてを再発見するニューラルネットワーク」から、
「既存知識をパラメータ状態として受け取り、未知部分だけを学習するニューラルネットワーク」への移行。**

---

## 14. 引き継ぎ時に最初に確認すること

次の担当者/次の会話では、最初に以下を読む。

1. この引き継ぎ書
2. `knowledge_to_weights_stage0.py`
3. `knowledge_to_weights_stage0_summary.csv`
4. `knowledge_to_weights_stage0_results.csv`
5. `knowledge_to_weights_stage0_curve.png`

そのうえで **Stage-1「0-step性能とoptimization advantageの分離」から再開する。**

現時点では新しいアイデアを増やすより、Stage-0の対立仮説を潰す方を優先する。

---

## 15. 現在の関連ファイル

- `knowledge_to_weights_stage0.py`
- `knowledge_to_weights_stage0_results.csv`
- `knowledge_to_weights_stage0_summary.csv`
- `knowledge_to_weights_stage0_curve.png`

