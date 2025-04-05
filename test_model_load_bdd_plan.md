# ModelLoad BDDテスト 最終計画書 (厳守)

## 概要

`image-annotator-lib` の `ModelLoad` クラスおよび関連するローダーの機能を、BDD (Behavior-Driven Development) スタイルでテストします。

## テスト対象

*   `image-annotator-lib/src/image_annotator_lib/core/model_factory.py` 内の `ModelLoad` クラス
*   `ModelLoad` が呼び出す**実際の**ローダークラス (`TransformersLoader`, `ONNXLoader` など)
*   **実際の**ユーティリティ関数 (`image_annotator_lib.core.utils` など)
*   **実際の**外部ライブラリ (`transformers`, `onnxruntime`, `tensorflow`, `torch`, `psutil` など)

## テストシナリオ

*   `tests/features/core/model_factory.feature` に記述されている全シナリオ。

## 実装ファイル

*   BDD ステップ定義: `tests/integration/test_model_factory_integration.py`
    *   **制約:** このファイル配置は一般的なBDDの慣習とは異なりますが、ユーザー指示に従います。

## リファクタリング方針・制約 (厳守)

1.  **モック不使用:** `unittest.mock` などによるモック化は**絶対に**行いません。テストは実ライブラリ、実ファイルシステム、実設定ファイル、場合によっては実モデルファイルに依存します。
2.  **ファイル構成:** ステップ定義は `tests/integration/test_model_factory_integration.py` に実装します。他のテスト関連ファイル（例: `tests/step_defs/`）は作成・編集しません。
3.  **ステップ定義の再利用:** 同じステップ文字列（例: `@then("以下のコンポーネントが生成される:")`）に対しては、テストコード内で**単一のステップ定義関数**を定義し、異なるシナリオから再利用されるようにします。
4.  **コード構成:** `tests/integration/test_model_factory_integration.py` ファイル内で、まず全ての `@given` 関数、次に全ての `@when` 関数、その次に全ての `@then` 関数をまとめて記述するようにコードの構成順序を整理します。
5.  **シナリオ指定:** 個別の `@scenario` デコレータは使用せず、代わりに `@scenarios("../features/core/model_factory.feature")` をファイルの適切な場所（通常はステップ定義の前）に一度だけ記述します。
6.  **コンテキスト共有禁止:** `@pytest.fixture` で定義した共有のコンテキスト辞書（例: `base_loader_context`）は使用しません。ステップ間で必要なデータは、`target_fixture` やステップ関数の引数を通じて渡します。
7.  **インポート文:** 全ての `import` 文をファイルの先頭にまとめて記述します。`try...except ImportError` でインポートを囲むことはしません。

## 実装手順 (厳守)

以下の手順を**シナリオごと**に繰り返します。

1.  **シナリオ選択:** `tests/features/core/model_factory.feature` から未実装のシナリオを一つ選択します。
2.  **ステップ定義実装:** 選択したシナリオに必要な `@given`, `@when`, `@then` ステップに対応する関数を、上記**リファクタリング方針・制約**に従って `tests/integration/test_model_factory_integration.py` に実装または追記します。既存のステップ定義は再利用します。
3.  **動作確認:** 実装したシナリオのテストが**実際に動作すること**を `uv run pytest tests/integration/test_model_factory_integration.py -k <シナリオ名>` のようなコマンドで確認します。（テスト実行には適切な環境設定が必要です）
4.  **次へ:** 動作確認ができたら、次の未実装シナリオを選択し、手順2に戻ります。全てのシナリオの実装と動作確認が完了するまで繰り返します。

## 懸念事項 (再掲)

*   **モック不使用による影響:** 実行時間、リソース消費、安定性、再現性、CI/CDへの影響、デバッグの困難さ。
*   **ファイル構成:** 非標準的な構成による可読性や将来的なメンテナンス性の問題。

## 関係性の図 (Mermaid - モック不使用, 最終方針反映)

```mermaid
graph LR
    A[model_factory.feature] -- defines scenarios --> B(test_model_factory_integration.py);
    B -- implements BDD steps (grouped by Given/When/Then) & uses @scenarios --> C{ModelLoad / Real Loaders};
    C -- interacts with --> D[Real External Libs];
    C -- interacts with --> E[Real File System / Config / Models];
    C -- interacts with --> F[Real psutil / System Info];
    subgraph "Test Code (Integration Dir)"
        direction LR
        B
        G[conftest.py (Real Config?)]
    end
    B -- uses --> G;
    B -- uses target_fixture & step args --> B;