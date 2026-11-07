# GitHub Public公開の手順と注意

## 推奨：2つの配布物を分ける

- **GitHubリポジトリ本体**：`NeuroCompiler_public_repo.zip`。主要報告書、コード、グラフ、集計表、ハッシュ索引を含む。Gitコミット可能な軽量版。
- **原本の保存**：`NeuroCompiler_complete_research_release.zip`。元の9つのZIPとsidecar資料を集めた大容量の調査アーカイブ。GitHub Releases等にバージョン固定して添付し、**Gitコミットしない**。

公開前に、ファイル内の著作権、共同研究者の権利、生成物のライセンス、ローカルパス・APIキー・個人情報の有無を最終確認してください。原本checkpointを公開する場合、利用・読込上のセキュリティ説明を添付してください。

## 空の公開リポジトリを作る場合

既存のGitHubリポジトリにREADMEを初期生成するとローカル初回pushの際に履歴競合が起こりえます。可能なら空のPublicリポジトリを作り、README等はこの配布物から追加します。

```bash
unzip NeuroCompiler_public_repo.zip
cd NeuroCompiler
python3 scripts/verify_manifest.py
git init -b main
git add .
git commit -m "docs: publish historical NeuroCompiler research archive"
git remote add origin https://github.com/YOUR_USERNAME/NeuroCompiler.git
git push -u origin main
```

既にGitHub側でREADMEを作った場合はcloneして内容をコピーし、差分を確認してcommitしてください。上記は新規空repo専用の例です。**YOUR_USERNAMEを自分のGitHubユーザー名へ変更する**こと。

## GitHub Releasesについて

Stage9Rの元ZIPは約131MBで、GitHubの通常のGitファイルサイズ制限100MBを超えます。GitHub Release assetとして大容量版を別添するか、共有先を別の永続保管場所に決めてください。Releaseのリンクを作成した後、`archives/source_archives.csv`のSHA-256でバイト同一性を確認します。

Git LFSを使う場合でも課金・転送量と履歴保存負担を検討し、最初は原本をGit管理から除外する方針を推奨します。

## ライセンス未選択

リポジトリをPublicにするだけでは、第三者が再利用・改変・再配布できるわけではありません。コードはApache-2.0、実験data/図はCC BY 4.0等が候補ですが、混在する権利を確認してから確定してください。このパッケージの`LICENSE.template`は**ライセンス本文でも許諾でもありません**。確定前に `LICENSE` ファイルを作成し、第三者依存物のNOTICEを補うこと。

## 公開前の必須確認

1. GitHubのオーナー・説明・Public設定を確認。
2. `python3 scripts/verify_manifest.py`を実行してパッケージ内の改変を検出。
3. 権利とプライバシーを確認（特にログ、外部取得素材、重みファイル）。
4. `docs/EVIDENCE_POLICY.md`に記載した「旧Stage9原本なし」「9Rは独立再実装」を削除しない。
5. READMEの性能主張を、引用元レポートの比較条件・seed数と併記する。
6. 公開後の変更は`docs/CORRECTIONS.md`で訂正理由を記録する。
