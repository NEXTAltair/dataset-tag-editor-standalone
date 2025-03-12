"""モデル関連の例外クラスを定義するモジュール"""


class ModelError(Exception):
    """モデル関連の基本例外クラス"""

    pass


class ModelNotFoundError(ModelError):
    """モデルが見つからない場合の例外"""

    pass


class ModelLoadError(ModelError):
    """モデルのロードに失敗した場合の例外"""

    pass


class InvalidModelConfigError(ModelError):
    """モデル設定が無効な場合の例外"""

    pass


class UnsupportedModelError(ModelError):
    """サポートされていないモデルタイプの場合の例外"""

    pass


class ModelExecutionError(ModelError):
    """モデル実行中にエラーが発生した場合の例外"""

    pass


class InvalidInputError(ModelError):
    """入力データが無効な場合の例外"""

    pass


class InvalidOutputError(ModelError):
    """出力データが無効な場合の例外"""

    pass


class OutOfMemoryError(Exception):
    """モデルのロードまたは推論中にメモリ不足が発生した場合に発生する例外"""

    pass