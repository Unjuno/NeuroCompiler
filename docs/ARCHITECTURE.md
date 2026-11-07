# 研究上の構成と実装上の到達点

## コンパイラの概念

```text
問題記述（数学式AST・関数descriptor・PDE/ODE条件）
                │
       Symbolic front-end
     （厳密正規化・同値性）
                │
          Semantic IR
                │
         Backend Router
        ┌───────┴────────┐
        │                │
Weight generator    Numerical solver
Direct / Meta      Spectral / Galerkin / Newton
        │                │
        └───────┬────────┘
                ▼
        解と残差・誤差検査
```

### 到達済み（実験実装）

- 固定の解析関数/方程式族からのタスク条件付き重み生成（Stages 0–4）。
- `+` と `×` の木構造による重み生成（Stage 5）。
- テストした可換半環rewritesの正規化と生成重みの一致（Stage 6）。
- 初期値を構造で満たすODEテンプレートと物理残差適応（Stages 7–8）。
- 限定ODE族のnumerical backends（Stage 8）。
- 固定正弦基底係数を生成する2-D PDE prototype（Stage 9R）。

### 未到達（構想）

- arbitrary AST / ODE / PDEを受理する汎用front-end。
- 不変量・連続PDE誤差の完全な形式保証。
- 問題族横断で検証済みの、自律的かつ校正済みsolver router。
- 公開されたproduction runtime、LLM向けLoRA compiler。

## 設計原則

1. **表現可能性と学習能力の分離**：固定基底空間での射影下限を計測し、係数生成誤差とは区別する。
2. **数式同値性はまずfront-endで処理**：証明できる法則はlossに任せない。
3. **弱いベースラインに勝っただけでは「学習の追加価値」としない**：最適化した未学習特徴、直接数値法、学習時間を合わせた対照を含める。
4. **失敗を保持**：matching失敗、発散、表示precision差、残差の盲点を消さない。
5. **意味と構文を区別**：同値rewriteから同じ入力を作る保証と、「意味を学習した」証拠は別。

各段階が共有の可変ABIを使っているわけではないため、全ステージのコードを単一importable libraryに見せかけない。再構築時に仕様を決めるべき対象はdescriptor schema、IR canonical form、dtype、target-network layout、backend contractと誤差証明境界である。
