# データとコードの来歴

すべての原本ZIPは `/mnt/data` の会話添付に由来します。元の各ZIPは整合性検査（`ZipFile.testzip()`）に合格しました。

- [`archives/source_archives.csv`](../archives/source_archives.csv) は9個の原本ZIPについてSHA256、byte数、収録ファイル数と保管先の推奨区分を記載しています。
- [`archives/curated_source_map.csv`](../archives/curated_source_map.csv) は公開Git版に選んだファイルごとの元ZIPとアーカイブ内パス、SHA256を記録しています。
- 補足ファイル（Stage1e、Stage3a/b、Stage3cコード、2026年10月の査読メモ）はsidecar原本由来と表示しています。

生データ・モデルを含む9原本アーカイブは公開Gitへ直接入れません。別途大容量配布用ZIPにまとめて、GitHub Releases等の場所へ手動添付する想定です。**現時点でRelease URLは設定されていません。**

## 原本の欠落と非対応

- 会話中で報告された旧Stage 9の最終生データ/学習コードは、現在の9ZIPおよびsidecarから確認できません。Stage 9Rは後から作られた限定PDEの独立再実装であり、旧Stage9と同一の実験ではありません。
- Stage1eは報告書、対照CSV、曲線をsidecarで保持しますが、元の実行スクリプトを確認できません。
- Stage3a/bには集計とStage3cへの発展記録が残りますが、完全な訓練パイプラインは回収できていません。

## 公開用Gitと原本保存の違い

公開用Gitに含むのはコード、レポート、主要集計CSV、図、ハッシュ索引です。raw task records/NPZ/binary checkpointsは二次配布物に保管。過去の学習結果を再計算したと言うには、原本の問題bankと実験コードを同じversionで実行し、学習seed単位の結果を比較する必要があります。
