# 解説: ModelLoad クラスのリファクタリングガイド

**注意:** このドキュメントは、開発初期段階で検討された `ModelLoad` クラス (現在の `src/image_annotator_lib/core/model_factory.py` 内の `ModelLoad` クラスに相当) のリファクタリングに関するガイドです。実際のコードは進化している可能性がありますが、設計の背景として参照してください。

## 当初の課題

初期の `ModelLoad` クラスの実装には以下の問題点が指摘されていました。

1.  **カプセル化の不足:** 内部状態を保持するクラス変数 (`_MODEL_STATES`, `_MODEL_LAST_USED`, `_LOADED_COMPONENTS`) が公開されており、他のクラス (主にフレームワーク別基底クラス) から直接アクセスされていました。これにより、意図しない状態変更のリスクや、将来的な内部実装の変更が困難になる可能性がありました。
2.  **インターフェースの欠如:** 内部状態に安全にアクセスするための適切な公開メソッド (アクセサ) が提供されていませんでした。
3.  **型安全性の問題:** 内部変数への直接アクセスにより、多くの箇所で型チェックエラーが発生し、それを抑制するために `type: ignore[attr-defined]` コメントが使用されていました。これはコードの可読性と信頼性を損なう要因でした。

## リファクタリング方針 (現在の設計思想)

当初のガイドでは、`ModelLoad` に状態を取得するアクセサメソッドを追加し、依存コード側でそれらのメソッドを呼び出すことが提案されていました。**しかし、このアプローチは現在の設計思想では採用されていません。**

現在の設計思想の核心は以下の通りです。

*   **アノテータークラス (`BaseAnnotator` のサブクラス) は `ModelLoad` の内部状態 (モデルがGPUにあるか、CPUにあるか、ロード済みかなど) を一切意識しない。**
*   **モデルの状態管理と、状態に基づいたロード/キャッシュ復元/CPU退避などの判断は、すべて `ModelLoad` クラスの責任とする。**

これにより、責務が明確に分離され、`ModelLoad` クラスの内部実装の変更がアノテータークラスに影響を与えにくくなります。

### 依存コードの修正 (現在の設計思想に基づく形)

アノテータークラス (特にフレームワーク別基底クラスの `__enter__` メソッド) は、`ModelLoad` に対して、必要なモデルコンポーネントを準備するように **依頼するだけ** です。`ModelLoad` 側が内部状態を確認し、必要に応じてロード、復元、キャッシュ取得、メモリ解放などの操作を行います。

**修正前 (例 - 内部状態への直接アクセス):**

    ```python
    # core/base.py (旧)
    # (概念を示すための簡略化された例)
    import time
    from image_annotator_lib.core.model_factory import ModelLoad # 仮のインポート

    class BaseAnnotator: # 仮の基底クラス
        def __init__(self, model_name, model_path, device):
            self.model_name = model_name
            self.model_path = model_path
            self.device = device
            self.components = {}

    class TransformersBaseAnnotator(BaseAnnotator):
        def __enter__(self):
            # ModelLoad の内部状態を直接参照・操作していた
            print(f"Checking state for {self.model_name}")
            current_state = ModelLoad._MODEL_STATES.get(self.model_name)
            print(f"Current state: {current_state}")
            ModelLoad._MODEL_LAST_USED[self.model_name] = time.time()
            self.components = ModelLoad._LOADED_COMPONENTS.get(self.model_name, {})
            print(f"Components obtained (directly): {bool(self.components)}")
            # ... さらに状態に応じたロード処理をアノテーター側で記述 ...
            print(f"Entering context for {self.model_name}")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            print(f"Exiting context for {self.model_name}")
            # 解放処理などもアノテーター側で状態を見て判断していた可能性
    ```

**修正後 (現在の設計思想に基づく形):**

アノテータークラスは、`ModelLoad` が提供するフレームワーク固有のロードメソッド (例: `load_transformer_components`) を呼び出します。このメソッドが状態管理と実際のロード/復元処理をすべて担当します。

    ```python
    # core/base.py (新)
    # (概念を示すための簡略化された例)
    from image_annotator_lib.core.model_factory import ModelLoad # 仮のインポート
    from image_annotator_lib.exceptions import ModelLoadError # 仮のインポート
    import logging

    logger = logging.getLogger(__name__)

    class BaseAnnotator: # 仮の基底クラス
         def __init__(self, model_name, model_path, device):
            self.model_name = model_name
            self.model_path = model_path
            self.device = device
            self.components = {}

    class TransformersBaseAnnotator(BaseAnnotator):
        def __enter__(self):
            # ModelLoad にコンポーネントの準備を依頼する
            # アノテーターはモデルの状態を知る必要はない
            logger.debug(f"Entering context for {self.model_name}, requesting components.")
            try:
                # (実際のメソッド名は model_factory.py の実装に依存)
                loaded_model = ModelLoad.load_transformer_components(
                    model_name=self.model_name,
                    model_path=self.model_path, # ロードに必要な情報を渡す
                    device=self.device
                    # その他の必要な設定 (例: torch_dtype) も渡す
                )
                if loaded_model:
                    self.components = loaded_model
                logger.debug(f"Components loaded for {self.model_name}.")
                # 最終使用時刻の更新も ModelLoad.load_xxx_components 内で処理される想定
            except Exception as e:
                # ModelLoad 側で発生したロード関連のエラーをハンドル
                logger.error(f"Failed to load model components for {self.model_name}: {e}", exc_info=True)
                # 適切な例外にラップして再送出
                raise ModelLoadError(f"Failed to load model components for {self.model_name}") from e
            return self # self を返すことで with ブロック内でインスタンスを使用できる

        def __exit__(self, exc_type, exc_val, exc_tb):
            # リソース解放やキャッシュへの退避も ModelLoad に依頼する
            # (実際のメソッド名は model_factory.py の実装に依存)
            logger.debug(f"Exiting context for {self.model_name}, releasing/caching components.")
            ModelLoad.release_or_cache_components(self.model_name, self.components) # 仮のメソッド名
            logger.debug(f"Components released/cached for {self.model_name}.")

    ```
このアプローチにより、アノテータークラスはモデルの状態管理の詳細から解放され、自身の本来の責務（前処理、推論実行、後処理）に集中できます。`ModelLoad` クラスはモデルの状態とリソース管理に関するすべてのロジックをカプセル化します。

## 実装上の注意点 (当初のガイドから)

当初のガイドで推奨された注意点は、公開メソッドを適切に設計する上で依然として重要です。

1.  **防御的コピー:** 辞書やリストなどのミュータブルなオブジェクトを返すメソッドでは、内部状態が外部から変更されるのを防ぐために、オブジェクトのコピーを返すようにします。
2.  **一貫性のある例外処理:** 公開メソッド内では、無効な入力に対するエラーハンドリングを一貫した方法で実装します。
3.  **型ヒントの徹底:** すべての公開メソッドに正確な型ヒントを付与します。
4.  **ドキュメンテーション:** 各公開メソッドに Docstring を追加し、その目的、引数、戻り値、発生しうる例外などを明確に記述します。

## 移行計画 (現在の設計思想に基づく)

1.  `ModelLoad` クラスに、状態管理ロジックを内包した高レベルな公開メソッド (例: `load_transformer_components`, `release_or_cache_components`) を実装または確認する。
2.  追加/修正したメソッドに型ヒントと Docstring を記述する。
3.  アノテータークラス (`core/base.py` 内) が `ModelLoad` の内部変数や低レベルな状態取得メソッドにアクセスしている箇所を特定し、新しい高レベルメソッドを使うように修正する。**アノテータークラスがモデルの状態を判断するロジックは削除する。**
4.  関連するユニットテストを追加・更新する。
5.  不要になった `type: ignore` コメントを削除する。

このリファクタリングにより、`ModelLoad` クラスのカプセル化が強化され、コードの保守性、信頼性、型安全性が向上することが期待されました。