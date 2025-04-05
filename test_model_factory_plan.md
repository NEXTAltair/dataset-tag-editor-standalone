# test_model_factory.py 修正・リファクタリング計画

## 1. 目的

`tests/step_defs/core/test_model_factory.py` のテスト実装におけるエラーを修正し、モックへの過度な依存を減らすことで、テストの信頼性を向上させる。また、Pytest BDD ステップ定義ガイドラインに準拠させる。

## 2. 現状の課題

1.  **テストエラー:** `StepDefinitionNotFoundError`, `TypeError`, `AttributeError`, `AssertionError` が多数発生。
2.  **過度なモック:** `ModelLoad` クラスのキャッシュ管理ロジックなどが、モック自体のテストになっている。
3.  **ステップ定義の不整合:** フィーチャーファイルとステップ定義のテキスト（デコレータ引数、関数引数）が一致していない。
4.  **ガイドラインへの準拠:** ステップ定義関数の命名規則 (`given_`, `when_`, `then_` プレフィックス) が未適用。

## 3. 修正・リファクタリング計画

### 3.1. エラー修正 (最優先)

-   **StepDefinitionNotFoundError / Fixture not found:**
    -   `@given`, `@when`, `@then` デコレータの引数を、フィーチャーファイルのテキストと**完全に**一致させる。特にコロン `:` や改行 `\n`、プレースホルダー `{}` の有無を確認する。
    -   テーブル引数を取るステップ (`Pipelineモデルの設定`, `CLIPモデルの設定`) では、デコレータ引数にコロンを含めず、関数定義で `table: str` 引数を受け取る。
    -   引数を取るステップでは `parsers.parse()` を使用し、引数を取らないステップでは使用しない。
-   **TypeError / AttributeError:**
    -   ローダー (`DummyLoader`, `TensorFlowLoader` など) の `__init__` に渡す引数を、基底クラス (`BaseModelLoader`) が受け付けるものに限定する (`model_path`, `format` など不要な引数を渡さない)。
    -   `patch` の対象パスを正しいライブラリモジュール (`transformers`, `onnxruntime`, `tensorflow` など) に修正する。
    -   `@patch` デコレータがフィクスチャ解決を妨げる場合は、関数内部での `with patch(...)` に切り替える。
    -   存在しない属性 (`model_path`, `_find_model_dir`) へのアクセスやモック化を修正する (`utils.load_file` のモックなどを活用)。
    -   `TensorFlowLoader` のテストで `loader.load()` の代わりに `loader.load_components(...)` を呼び出し、モック検証を修正する。
-   **AssertionError (`test_キャッシュ制御`):**
    -   `mock_model_load_statics` フィクスチャ内のキャッシュ解放ロジック (`mock_clear_cache_if_needed`) を修正し、期待通りに動作するようにする。

### 3.2. モック範囲の見直し & 実コードテスト (リファクタリング)

-   **`ModelLoad` のテスト (キャッシュ制御、CUDA管理):**
    -   `mock_model_load_statics` フィクスチャを修正・改名 (例: `setup_model_load_state`)。
    -   `monkeypatch` を使用して `ModelLoad` のクラス変数 (`_MODEL_STATES`, `_MEMORY_USAGE` など) をテストごとに初期化/設定する。
    -   外部依存 (`time.time`, `psutil.virtual_memory` など) のみをモックする。
    -   実際の `ModelLoad` の静的メソッド (`cache_to_main_memory`, `restore_model_to_cuda` など) を呼び出し、クラス変数の状態変化をアサートする。
-   **各ローダー (`TransformersLoader` など) のテスト:**
    -   時間のかかる外部ライブラリ呼び出し (`from_pretrained` など) と `utils` 関数はモックする。
    -   ローダー自身の初期化、引数処理、モック関数呼び出し、結果格納などのロジックは実コードでテストする。
-   **`BaseModelLoader` のテスト:**
    -   `DummyLoader` の使用をやめ、実際のサブクラス (例: `TransformersLoader`) を使って共通属性の初期化を検証する。
-   **`@given` ステップのデータ設定:**
    -   フィーチャーファイルのパラメータを基本とする。
    -   ファイル/ディレクトリ構造が必要なテストでは `tmp_path` フィクスチャを利用して一時ファイル/ディレクトリを作成し、そのパスを使用する。

### 3.3. ガイドライン準拠

-   すべてのステップ定義関数名に `given_`, `when_`, `then_` プレフィックスを追加する。

## 4. 進め方

1.  まずステップ 3.1 のエラー修正を行い、テストが通る状態を目指す。
2.  次にステップ 3.2 のリファクタリングを行う。
3.  最後にステップ 3.3 のガイドライン準拠を行う。

## 5. Mermaid ダイアグラム (テスト構造の概要)

```mermaid
graph TD
    A[Feature File: model_factory.feature] --> B(Step Definitions: test_model_factory.py);
    B --> C{pytest-bdd};
    C --> D[Fixtures: model_factory_context, tmp_path];
    B -- Uses --> D;
    B -- Uses --> E(Mocking: patch, MagicMock, monkeypatch);
    B -- Tests --> F(Target Code: model_factory.py);
    F -- Depends on --> G[External Libs];
    F -- Depends on --> H[Internal Utils];
    F -- Depends on --> I[Config Files: models.toml];
    F -- Depends on --> J[File System];
    E -- Mocks --> G;
    E -- Mocks --> H;
    E -- Mocks/Simulates --> J[via tmp_path];
    E -- Controls --> F[Internal State via monkeypatch];