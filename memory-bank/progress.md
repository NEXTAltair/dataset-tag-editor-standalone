# memory-bank/progress.md (Updated: 2025/04/05)

## 開発環境の注意点 (パス情報)

ファイルパスの指定ミスを防ぐため、主要なパス情報を以下に記載します。

- **プロジェクトルート:** `dataset-tag-editor-standalone`
  - すべてのファイル操作 (read_file, write_to_file など) は、原則としてこのディレクトリからの相対パスで指定します。
- **`image-annotator-lib` パッケージルート:** `image-annotator-lib/src/image_annotator_lib`
  - ライブラリ内部のモジュールを参照・編集する際の基点となります。
  - 例: `image-annotator-lib/src/image_annotator_lib/core/base.py`

---

## プロジェクトの現状サマリー

`scorer_wrapper_lib` と `tagger_wrapper_lib` を統合した `image-annotator-lib` の開発状況。

### 完了済みの主要な作業

1.  **設計アウトライン定義**:
    - 統合ライブラリの基本構造、クラス階層、主要コンポーネントの方針を定義。
2.  **コアモジュールの実装**:
    - `core/base.py`: `BaseAnnotator` およびフレームワーク別基底クラス (`ONNXBaseAnnotator`, `TransformersBaseAnnotator`, `TensorflowBaseAnnotator`, `ClipBaseAnnotator`, `PipelineBaseAnnotator`) を実装。`AnnotationResult` TypedDict を定義。
    - `core/model_factory.py`: `ModelLoad` クラスを実装。ONNX, Transformers, TensorFlow, Pipeline, CLIP モデルのロード/キャッシュ/解放ロジックを実装。
    - `core/registry.py`: `ModelRegistry` を実装。設定ファイルと動的クラス検出に基づきモデルクラスを登録・取得する関数 (`register_annotators`, `get_annotator_class`, `list_available_annotators`) を実装。
    - `core/utils.py`: 設定ファイル (`config/annotator_config.toml`) 読み込み (`load_model_config`)、ファイルダウンロード/キャッシュユーティリティ、ロガー設定 (`setup_logger`) などを実装。
    - `exceptions/errors.py`: カスタム例外クラス (`AnnotatorError`, `ModelLoadError`, `ModelNotFoundError`, `OutOfMemoryError` など) を定義。
    - `api.py`: 主要 API 関数 `annotate` (旧 `evaluate`) を実装。pHash ベースの結果集約ロジックを含む。
3.  **モデルクラスの移植**:
    - 各種 Tagger (ONNX, Transformers, TensorFlow) および Scorer (Aesthetic, CLIP) モデルクラスを `models/` ディレクトリに移植し、新しいクラス階層 (`BaseAnnotator` 継承) に適合。
4.  **コードのリネーム**:
    - 主要 API 関数名を `evaluate` から `annotate` に変更。
    - 設定ファイル名を `models.toml` から `annotator_config.toml` に変更。
5.  **ドキュメント整理**:
    - `docs/` ディレクトリ内のドキュメントを Diátaxis フレームワークに基づいて整理・統合。
    - `README.md` を更新。
    - `TUTORIALS`, `HOW_TO_GUIDES`, `REFERENCE`, `EXPLANATION` ディレクトリを作成し、関連ドキュメントを移植・作成。
    - 古いドキュメントファイルを削除。
6.  **Memory Bank 更新**:
    - `memory-bank/productContext.md`, `memory-bank/decisionLog.md`, `memory-bank/progress.md` を最新情報に更新。
    - `memory-bank/activeContext.md` を整理されたドキュメントに基づいて更新 (2025-04-05)。`ModelLoad` の二階層構造に関する情報を追記。
    - `memory-bank/productContext.md` にパス情報と重要なコーディングルールを追記 (2025-04-05)。
    - `memory-bank/decisionLog.md` にドキュメント整理とルール明確化に関する決定事項を追記 (2025-04-05)。

### 残りの作業と確認事項

1.  **テスト拡充**:

    - **単体テスト**:
      - `ModelLoad` のキャッシュ戦略(LRU、CPU 退避、メモリ逼迫時の挙動)を検証するテストケースを追加。
      - 各フレームワーク別ローダー (`TransformersLoader`, `ONNXLoader` 等) の正常系・異常系テストを追加。
      - `registry.py` の動的クラス検出・登録ロジックのテストを追加。
      - 各具象モデルクラス (`models/` 配下、`ImageRewardScorer` を除く) の `_generate_tags` 等の主要メソッドに対するテストを追加・拡充。
    - **結合テスト (BDD 含む)**:
      - `api.annotate` 関数のテストシナリオを拡充 (複数モデル同時実行、pHash 生成失敗ケース、エラーハンドリングなど)。
      - 各モデルのエラーハンドリング (特に OOM) が `annotate` 関数レベルで適切に処理され、結果に反映されるか検証するシナリオを追加。
      - 設定ファイル (`annotator_config.toml`) の様々なパターン (オプション指定有無など) に対するテストを追加。
    - **カバレッジ**: `pytest --cov` を実行し、テストカバレッジを確認・向上させる。

2.  **モデル実装の最終確認**:

    - 各具象モデルクラス (`models/` 配下、`ImageRewardScorer` を除く) のデフォルトパラメータ(閾値など)を確認し、必要であれば調整・ドキュメント (`REFERENCE/models.md` 等) に追記。
    - 特に `CafePredictor` や `AestheticShadowV1` など、Scorer 系のモデルの出力形式 (`tags` または `formatted_output`) が `AnnotationResult` の定義と整合しているか再確認。

3.  **ドキュメント最終化**:

    - `docs_image-annotator-lib` 内の各ドキュメント (API リファレンス、設定ガイド、モデル説明など) の内容を、最新のコード実装と完全に一致するように最終レビュー。
    - 全公開クラス・メソッド・関数の Docstring の網羅性と正確性を確認・修正。
    - 必要に応じてアーキテクチャ図などを `EXPLANATION/` に追加・更新。
    - 日本語 README (`README-JP.md`) を作成または更新。

4.  **依存関係の最終整理**:

    - `image-annotator-lib/pyproject.toml` の依存関係 (`dependencies`, `dev-dependencies`) を最終確認し、不要なライブラリがあれば削除。バージョン指定が適切か確認。

5.  **実環境での動作確認**:

    - ライブラリを実際に使用する環境 (例: stable-diffusion-webui 拡張機能) で `annotate` 関数を呼び出し、主要なモデルが問題なく動作するか確認。パフォーマンスやメモリ使用量についても簡易的に確認。

6.  **静的解析エラーの完全解消**:

    - `ruff check` および `mypy` を実行し、報告されるエラーや警告が完全に解消されていることを確認。(# type: ignore や # noqa が残っていないか確認)

7.  **設定ファイル (`annotator_config.toml`) の構造改善検討 (TODO)**:
    - 現在の `annotator_config.toml` の構造は読みにくいため、将来的に改善を検討する。具体的な改善案については別途相談する。

# 進捗状況

## 現在の作業項目

### ModelLoad 改善 (2024-04-02 -> 2025-04-05)

- [x] 設計変更の決定と文書化
- [x] 基底ローダークラス(BaseModelLoader)の実装
- [x] 具象ローダークラスの実装
  - [x] TransformersLoader
  - [x] ONNXLoader
  - [x] CLIPLoader
  - [x] TensorFlowLoader (追加)
  - [x] TransformersPipelineLoader (追加)
- [ ] テストケースの更新
- [ ] 既存コードの移行

### 優先度の高いタスク (更新: 2025-04-05)

1.  テスト拡充 (単体・結合・カバレッジ)
2.  モデル実装の最終確認 (`ImageRewardScorer` 除く)
3.  ドキュメント最終化 (コードとの整合性、Docstring、README-JP)
4.  依存関係の最終整理
5.  実環境での動作確認
6.  静的解析エラーの完全解消
7.  設定ファイル (`annotator_config.toml`) の構造改善検討 (将来対応)

## 完了した作業

### Memory Bank 更新 (2025-04-05)

- `activeContext.md` を整理されたドキュメントに基づいて更新。`ModelLoad` の二階層構造に関する情報を追記。
- `progress.md` にパス情報を追記。
- `productContext.md` にパス情報と重要なコーディングルールを追記。
- `decisionLog.md` にドキュメント整理とルール明確化に関する決定事項を追記。
- `progress.md` の TODO リストを具体化し更新。

### ModelLoad 設計 (2024-04-02)

- 二階層構造の設計決定
- 設計文書の作成
- コンポーネント要件の定義
