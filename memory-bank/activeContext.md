# 現在の開発コンテキスト (更新日: 2025-04-05)

このドキュメントは、最近整理された `docs_image-annotator-lib` および `.cursor/rules` 内のドキュメントに基づき、現在の開発フェーズに関連する主要な概念、ルール、およびコンテキストを要約したものです。

## 1. `image-annotator-lib` のコアコンセプト

- **統一 API:** 主要なエントリーポイントは `src/image_annotator_lib/api.py` 内の `annotate(images: list[Image.Image], model_names: list[str])` 関数です。
- **結果形式:** `annotate` 関数の戻り値は `{phash: {model_name: {"tags": list[str], "formatted_output": Any, "error": Optional[str]}}}` 形式の辞書です。
  - `phash` (知覚ハッシュ) が結果を入力画像に紐付けます。
  - pHash 計算に失敗した場合は `unknown_image_{index}` が使用されます。
- **クラス階層 (3 層):**
  1.  **`BaseAnnotator` (`core/base.py`):**
      - 共通インターフェース (抽象メソッド) を定義します。
      - **共通の `predict()` メソッド** (チャンク処理、pHash、エラーハンドリング、結果生成を含む) を提供します。**`predict()` はオーバーライド禁止です。**
      - サブクラスはヘルパーメソッド (`_preprocess_images`, `_run_inference`, `_format_predictions`, `_generate_tags`) を実装する必要があります。
  2.  **フレームワーク別基底クラス (`core/base.py`, 例: `ONNXBaseAnnotator`):**
      - `BaseAnnotator` を継承します。
      - フレームワーク固有のロジック (モデルのロード/アンロード (`__enter__`/`__exit__`)、ヘルパーメソッドのデフォルト実装) を実装します。
  3.  **具象モデルクラス (`models/`):**
      - フレームワーク別基底クラスを継承します。
      - モデル固有のロジック、特に **`_generate_tags()`** を実装します。必要に応じて他のヘルパーメソッドをオーバーライドします。
- **主要コンポーネント:**
  - **`ModelLoad` (`core/model_factory.py`):** モデルのロード、キャッシュ管理 (LRU、CPU 退避)、メモリ推定、リソース解放を担当します。**二階層構造**を採用しており、`ModelLoad` クラスの静的メソッドが、共通のキャッシュ管理ロジックを持つ `BaseModelLoader` を継承したフレームワーク別ローダークラス (例: `TransformersLoader`, `ONNXLoader`) を呼び出します。
  - **`registry.py` (`core/registry.py`):** モジュールレベルの関数と辞書が、`annotator_config.toml` に基づいて `models/` 内の具象モデルクラスを動的に検出し登録します。`list_available_annotators()` および `get_annotator_class()` を提供します。
  - **`config.py` (`core/config.py`):** 設定管理クラス `ModelConfigRegistry` とその共有インスタンス `config_registry` を定義します。`config_registry` は設定ファイル (`annotator_config.toml`) の内容全体を保持し、`get(model_name, key, default)` メソッドを通じて設定値へのアクセスを提供します。
  - **`utils.py` (`core/utils.py`):** ロギング設定 (`setup_logger`) と、設定管理以外の汎用的なユーティリティ関数 (ファイル I/O、ネットワーク処理、画像処理など) を提供します。

## 2. 主要な開発ルール

詳細は `.cursor/rules/` を参照してください。

- **コーディング (`coding-rules.mdc`):**
  - 環境: Python 3.12, Windows 11, 依存関係管理に `uv` を使用。
  - スタイル: PEP 8 準拠、パス操作には `pathlib` を使用、意味のある命名規則 (`snake_case` / `CamelCase`、汎用的な `*er` 名は避ける)。リスト内包表記の複雑さを制限。
  - **型ヒント:** モダンな型 (`list`, `dict`, `TypedDict`, `Self`) を使用。**`Any` の使用は厳禁**。すべての関数/メソッドに型ヒントを付与。**`# type: ignore` や `# noqa` は使用禁止**。`Optional` は使用しない。
  - **記号:** **半角文字のみ**を使用。全角文字/記号は使用禁止。
  - エラーハンドリング: 特定の予期されるエラー (`FileNotFoundError`, `ValueError`, `KeyError`, `OutOfMemoryError`) のみを捕捉。広範な `except Exception` は避ける。予期しないエラーはログに記録し、伝播させるか停止する。情報を含むエラーメッセージを記述。
- **カプセル化 (`encapsulation-rules.mdc`):**
  - **厳禁:** 他クラスの内部変数 (`_` プレフィックス) への直接アクセス。
  - 「Tell, Don't Ask」原則に従う。公開インターフェースを最小化 (YAGNI)。
  - 内部状態は非公開。読み取り専用アクセスが必要な場合は `@property` を使用 (イミュータブルな型を返す、防御的コピーは原則禁止)。
  - **ゲッター/セッターメソッドは禁止。**
- **ドキュメンテーション (`project-documentation-rules.mdc`):**
  - すべての関数/メソッドに Google スタイルの docstring (Args, Returns, Raises) を追加。
  - モジュールの目的、コンポーネント、依存関係を説明するモジュールレベルコメントを追加。
  - 明確な日本語の実装コメントを記述。
  - **コード追加/変更時には Todo Tree タグ (`TODO`, `FIXME`, `OPTIMIZE`, `BUG`, `HACK`, `XXX`, `[ ]`, `[x]`) を使用**し、意図や課題を明確化。
  - コード変更時には関連ドキュメント (docstring, `docs_image-annotator-lib/`) も更新。
- **ディレクトリ構造 (`directory-structure-rules.mdc`):**
  - ライブラリソース: `src/image_annotator_lib/` (`core/`, `exceptions/`, `models/`, `api.py`)。
  - テスト: プロジェクトルート `tests/` (`unit/`, `integration/`, `features/`, `step_defs/`, `resources/`)。ユニットテストは `src/` 構造を反映。
- **`image-annotator-lib` 固有ルール (`image-annotator-lib.mdc`):**
  - クラス責務とオーバーライドルール (特に **`predict()` のオーバーライド禁止**) を再確認。

## 3. ドキュメント構成概要

- **`docs_image-annotator-lib/`:** ライブラリの包括的なドキュメント。
  - `getting_started.md`: 基本的な使い方チュートリアル。
  - `developer_guide.md`: モデル追加、テスト実行、ロギング設定方法。
  - `EXPLANATION/`: 設計原則、アーキテクチャ、決定事項、履歴。
  - `REFERENCE/`: API 詳細、設定、モデル仕様。
- **`.cursor/rules/`:** AI が遵守すべき開発ルール。
  - `coding-rules.mdc`
  - `directory-structure-rules.mdc`
  - `encapsulation-rules.mdc`
  - `project-documentation-rules.mdc`
  - `image-annotator-lib/image-annotator-lib.mdc` (ライブラリ固有実装ルール)
  - `test_rules/` (テスト固有ルール)
