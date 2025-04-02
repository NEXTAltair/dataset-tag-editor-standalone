# ハウツーガイド: ロギングの設定

このガイドでは、`image-annotator-lib` におけるロガーの設定方法と利用方法について説明します。適切なロギング設定は、デバッグや動作状況の把握に役立ちます。

## 1. 基本的なロガーの利用

ライブラリ内の各モジュールでは、Python 標準の `logging` モジュールを使用することが推奨されます。モジュールレベルでロガーを取得するのが一般的です。

```python
import logging

# モジュールレベルでロガーを取得
# __name__ を使うことで、ロガー名がモジュールのフルパス (例: 'image_annotator_lib.models.tagger_onnx') になる
logger = logging.getLogger(__name__)

# ログメッセージの出力例
logger.info("これは情報メッセージです。")
logger.debug("これはデバッグメッセージです。")
logger.warning("これは警告メッセージです。")
```

## 2. ロガーの初期設定 (`setup_logger`)

ライブラリ全体で使用するロガーの基本的な設定 (ログレベル、フォーマット、出力先など) は、`core/utils.py` 内の `setup_logger` 関数で行われています。この関数は、ライブラリの初期化時などに内部的に呼び出されることがあります。

通常、ライブラリの利用者が直接 `setup_logger` を呼び出す必要はありませんが、ライブラリの挙動を理解する上で重要です。

**`setup_logger` の主な機能:**

*   指定された名前でロガーを取得または作成します。
*   ログレベルを設定します (デフォルトは `logging.INFO`)。
*   ログフォーマットを設定します (`%(asctime)s - %(name)s - %(levelname)s - %(message)s`)。
*   標準出力 (コンソール) とログファイル (`logs/image_annotator_lib.log`) の両方にログを出力するハンドラを設定します。
    *   ログファイル用のディレクトリ (`logs/`) が存在しない場合は自動的に作成されます。
*   ハンドラの重複設定を防ぎます。

**`core/utils.py` 内の `setup_logger` の定義 (抜粋):**

```python
# src/image_annotator_lib/core/utils.py

import logging
from pathlib import Path

LOG_FILE = Path("logs/image_annotator_lib.log") # ログファイルパス

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers: # ハンドラが未設定の場合のみ追加
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # 標準出力ハンドラ
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # ファイルハンドラ
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8") # UTF-8で書き込み
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
```

## 3. ログレベルの制御

ログレベルを変更することで、出力されるログの詳細度を調整できます。

*   **`logging.DEBUG`**: 最も詳細な情報。問題の診断時に役立ちます。
*   **`logging.INFO`**: 通常の動作状況を示す情報。
*   **`logging.WARNING`**: 予期しない事態や、将来問題を引き起こす可能性のある状況を示す警告。
*   **`logging.ERROR`**: より深刻な問題により、プログラムが一部の機能を実行できなかったことを示すエラー。
*   **`logging.CRITICAL`**: プログラム自体の継続的な実行が不可能になるほどの深刻なエラー。

ライブラリ利用時に、特定のモジュールや全体のログレベルを変更したい場合は、標準の `logging` モジュールを使って設定できます。

```python
import logging

# 例: 'image_annotator_lib' 全体のログレベルを DEBUG に設定
logging.getLogger('image_annotator_lib').setLevel(logging.DEBUG)

# 例: 特定のモデルモジュールのログレベルを DEBUG に設定
logging.getLogger('image_annotator_lib.models.tagger_onnx').setLevel(logging.DEBUG)
```

## 4. クラス内でのロギング

クラス内でロギングを行う場合も、モジュールレベルと同様に `logging.getLogger(__name__)` を使用するか、クラス名をロガー名に含めることで、ログの発生源を特定しやすくできます。

```python
import logging

logger = logging.getLogger(__name__) # モジュールレベルロガー

class MyAnnotator:
    def __init__(self, model_name: str):
        # クラス名をロガー名に含める場合 (任意)
        # self.class_logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # self.class_logger.info(f"Initializing MyAnnotator for {model_name}")
        logger.info(f"Initializing MyAnnotator for {model_name}")
        self.model_name = model_name

    def some_method(self):
        logger.debug(f"Executing some_method in MyAnnotator ({self.model_name})")
        # self.class_logger.debug("Executing some_method") # クラスロガーを使う場合
        try:
            # 何らかの処理
            pass
        except Exception as e:
            logger.error(f"Error in some_method: {e}", exc_info=True) # エラー発生時は詳細情報も記録
```

## まとめ

*   ライブラリ内のログは標準の `logging` モジュールを使用して出力されます。
*   基本的な設定は `core/utils.py` の `setup_logger` で行われ、コンソールとログファイル (`logs/image_annotator_lib.log`) に出力されます。
*   ログレベルは標準の `logging` モジュールを通じて外部から制御可能です。
*   `logging.getLogger(__name__)` を使用することで、ログの発生源を特定しやすくなります。

**注意:** このドキュメントでは、ユーザーの指示に基づき、設定ファイル名を `annotator_config.toml` と記述していますが、現在の実際のコード実装では `models.toml` となっています。