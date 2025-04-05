# 解説: ModelLoad クラスの設計

このドキュメントは、`src/image_annotator_lib/core/model_factory.py` 内の `ModelLoad` クラスの設計と、関連するクラスとの連携について解説します。

## 設計に至った背景 (課題)

開発初期の `ModelLoad` クラスの実装には以下の問題点がありました。

1.  **カプセル化の不足:** 内部状態を保持するクラス変数 (`_MODEL_STATES`, `_MODEL_LAST_USED`, `_LOADED_COMPONENTS`) が公開されており、他のクラス (主にフレームワーク別基底クラス) から直接アクセスされていました。これにより、意図しない状態変更のリスクや、将来的な内部実装の変更が困難になる可能性がありました。
2.  **インターフェースの欠如:** 内部状態に安全にアクセスするための適切な公開メソッドが提供されていませんでした。
3.  **型安全性の問題:** 内部変数への直接アクセスにより、多くの箇所で型チェックエラーが発生し、それを抑制するために `type: ignore[attr-defined]` コメントが使用されていました。これはコードの可読性と信頼性を損なう要因でした。

これらの課題を解決するため、現在の設計が採用されました。

## 設計思想 (現在の設計)

現在の設計思想の核心は以下の通りです。

- **アノテータークラス (`BaseAnnotator` のサブクラス) は `ModelLoad` の内部状態 (モデルが GPU にあるか、CPU にあるか、ロード済みかなど) を一切意識しない。**
- **モデルの状態管理と、状態に基づいたロード/キャッシュ復元/CPU 退避などの判断は、すべて `ModelLoad` クラスの責任とする。**

これにより、責務が明確に分離され、`ModelLoad` クラスの内部実装の変更がアノテータークラスに影響を与えにくくなります。

### アノテータークラスと ModelLoad の連携

アノテータークラス (特にフレームワーク別基底クラスの `__enter__` メソッド) は、`ModelLoad` に対して、必要なモデルコンポーネントを準備するように **依頼するだけ** です。`ModelLoad` 側が内部状態を確認し、必要に応じてロード、復元、キャッシュ取得、メモリ解放などの操作を **自律的に** 行います。

**連携例 (概念):**

アノテータークラスは、`ModelLoad` が提供するフレームワーク固有の高レベルなロードメソッド (例: `load_transformer_components`) を呼び出します。このメソッドが状態管理と実際のロード/復元処理をすべて担当します。

    ```python
    # core/base.py (概念を示すための簡略化された例)
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
                # ModelLoadが提供する高レベルメソッドを呼び出す
                # (実際のメソッド名は model_factory.py の実装に依存)
                loaded_model = ModelLoad.load_transformer_components(
                    model_name=self.model_name,
                    model_path=self.model_path, # ロードに必要な情報を渡す
                    device=self.device
                    # その他の必要な設定 (例: torch_dtype) も渡す
                )
                # ModelLoadが返したコンポーネントを使用
                if loaded_model:
                    self.components = loaded_model
                logger.debug(f"Components loaded for {self.model_name}.")
                # 状態管理や最終使用時刻の更新は ModelLoad 内で処理される
            except Exception as e:
                # ModelLoad 側で発生したロード関連のエラーをハンドル
                logger.error(f"Failed to load model components for {self.model_name}: {e}", exc_info=True)
                # 適切な例外にラップして再送出
                raise ModelLoadError(f"Failed to load model components for {self.model_name}") from e
            return self # self を返すことで with ブロック内でインスタンスを使用できる

        def __exit__(self, exc_type, exc_val, exc_tb):
            # リソース解放やキャッシュへの退避も ModelLoad に依頼する
            # ModelLoadが提供する高レベルメソッドを呼び出す
            # (実際のメソッド名は model_factory.py の実装に依存)
            logger.debug(f"Exiting context for {self.model_name}, releasing/caching components.")
            ModelLoad.release_or_cache_components(self.model_name, self.components) # 仮のメソッド名
            logger.debug(f"Components released/cached for {self.model_name}.")

    ```

このアプローチにより、アノテータークラスはモデルの状態管理の詳細から解放され、自身の本来の責務（前処理、推論実行、後処理）に集中できます。`ModelLoad` クラスはモデルの状態とリソース管理に関するすべてのロジックをカプセル化します。

## 実装上の注意点

`ModelLoad` クラスおよびその公開メソッドを実装・変更する際には、以下の点に注意することが重要です。

1.  **内部状態の非公開:** クラスの内部状態 (特に `list` や `dict` のようなミュータブルなオブジェクト) を外部に直接返さないように設計します。状態を公開する必要がある場合は、`@property` を使用し、必要最小限の情報をイミュータブルな形式 (例: `tuple`、単純な値) で返すか、状態を返す代わりにその状態に基づいた操作を行うメソッド (Tell, Don't Ask の原則) を提供することを検討します。防御的コピー (`copy()`, `deepcopy()`) の使用も原則として避けます。
2.  **一貫性のある例外処理:** 公開メソッド内では、無効な入力やロード中のエラーなどに対するエラーハンドリングを一貫した方法で実装し、ライブラリ固有の適切な例外 (`ModelLoadError` など) を送出します。
3.  **型ヒントの徹底:** すべての公開メソッドに正確な型ヒントを付与します。
4.  **ドキュメンテーション:** 各公開メソッドに Docstring を追加し、その目的、引数、戻り値、発生しうる例外などを明確に記述します。

この設計により、`ModelLoad` クラスのカプセル化が強化され、コードの保守性、信頼性、型安全性が向上しています。
