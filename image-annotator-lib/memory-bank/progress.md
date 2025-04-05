# memory-bank/progress.md (Updated: 2025/04/02)

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

### 残りの作業 (想定)

1.  **テスト**:
    - 単体テスト、結合テスト (BDD 含む) の作成・実行。
    - 既存テストの修正・拡充 (特に API 変更、pHash ベースの戻り値に対応)。
    - カバレッジの確認。
2.  **ドキュメント最終化**:
    - 各ドキュメントの内容をコード実装と完全に一致するように最終レビュー。
    - Docstring の網羅性と正確性の確認。
    - 必要に応じて図 (アーキテクチャ図など) を追加。
    - 日本語 README (`README-JP.md`) の作成または更新。
3.  **依存関係の最終整理**:
    - `pyproject.toml` の依存関係を最終確認し、不要なものを削除。
4.  **動作確認**:
    - ライブラリを実際に使用する環境 (例: stable-diffusion-webui 拡張機能) で動作させ、問題がないか確認。
5.  **モデル実装の確認**:
    - `ImageRewardScorer` の実装状況の再確認とドキュメント修正。
    - 各モデルのデフォルトパラメータ（閾値など）の確認とドキュメント追記。

### 既知の問題・注意点 (要確認)

- Mypy エラーが完全に解消されているか確認が必要。
- 各モデルのエラーハンドリング (特に OOM) が適切に行われているか、テストを通じて確認が必要。
- `ModelLoad` のキャッシュ戦略 (特にメモリ逼迫時の挙動) の十分なテストが必要。

# 進捗状況

## 現在の作業項目

### ModelLoad 改善 (2024-04-02)

- [x] 設計変更の決定と文書化
- [ ] 基底ローダークラス（BaseModelLoader）の実装
- [ ] 具象ローダークラスの実装
  - [ ] TransformersLoader
  - [ ] ONNXLoader
  - [ ] CLIPLoader
- [ ] テストケースの更新
- [ ] 既存コードの移行

### 優先度の高いタスク

1. ModelLoad の階層構造実装
2. BDD テストの更新
3. 既存機能の移行テスト

## 完了した作業

### ModelLoad 設計 (2024-04-02)

- 二階層構造の設計決定
- 設計文書の作成
- コンポーネント要件の定義
