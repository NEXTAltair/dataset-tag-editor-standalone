# BDDテスト実装 参照ファイル・ディレクトリ (model_factory)

**重要:** このメモは、`model_factory.feature` のBDDテスト実装時に常に参照すること。

**プロジェクトルート:** `H:\Git\dataset-tag-editor-standalone-1` (全てのパスの基点)

**主要ディレクトリ/ファイル:**

1.  **テスト対象コード:**
    *   パス: `image-annotator-lib/src/image_annotator_lib/core/model_factory.py`
    *   内容: `ModelLoad` クラス、各種ローダークラス
    *   役割: テスト対象のビジネスロジック

2.  **フィーチャーファイル (仕様):**
    *   パス: `tests/features/core/model_factory.feature`
    *   内容: テストすべきシナリオ (Given/When/Then)
    *   役割: テストの仕様書

3.  **ステップ定義ファイル (実装場所):**
    *   パス: `tests/integration/test_model_factory_integration.py`
    *   内容: フィーチャーファイルのステップに対応するPythonコード
    *   役割: テストコードの実装先 (**ユーザー指定の場所**)

4.  **共通フィクスチャ:**
    *   パス: `tests/conftest.py` (プロジェクトルート直下の `tests` 内)
    *   内容: テスト全体で使用される可能性のある共通のセットアップコード
    *   役割: テストの前提条件準備

5.  **設定ファイル:**
    *   パス: `config/annotator_config.toml` (プロジェクトルート直下)
    *   内容: アノテーター全体の設定
    *   役割: 設定関連のテストで参照される可能性 (**`models.toml` は使用しない**)

**注意点:**

*   **ファイル読み込み:** `read_file` ツールを使用する際は、混乱を避けるため**プロジェクトルートからの相対パス**を常に使用する。 (例: `image-annotator-lib/src/...`, `tests/features/...`)
*   テストコード (`tests/integration/test_model_factory_integration.py`) 内からフィーチャーファイルを参照する際の相対パスは `'../features/core/model_factory.feature'` となる。
*   テスト実行コマンドは**プロジェクトルート** (`H:\Git\dataset-tag-editor-standalone-1`) から実行する。
    *   例: `uv run pytest tests/integration/test_model_factory_integration.py -k "シナリオ名"`
*   `image-annotator-lib` 内の `tests` ディレクトリは、今回の実装では**使用しない**。
*   他のテストファイル (`tests/integration/test_registry_integration.py` など) のフィーチャーファイルパス指定エラーは依然として存在する可能性がある。