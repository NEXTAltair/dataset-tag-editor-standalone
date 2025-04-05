"""ライブラリ固有のカスタム例外クラス。"""


class AnnotatorError(Exception):
    """image-annotator-lib の基底例外クラス。

    ライブラリ内で発生する特定の運用エラーを示すために使用されます。
    """

    pass


class ModelLoadError(AnnotatorError):
    """モデルのロード中にエラーが発生した場合の例外。

    モデルファイルの欠損、フォーマット不正、依存関係の問題などが原因で発生します。

    Attributes:
        message: エラーの詳細メッセージ。
    """

    def __init__(self, message: str):
        """ModelLoadError を初期化します。

        Args:
            message: エラーの詳細メッセージ。
        """
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return f"モデルロードエラー: {self.message}"


class ModelNotFoundError(AnnotatorError):
    """要求されたモデルがレジストリまたは設定に見つからない場合の例外。

    Attributes:
        model_name: 見つからなかったモデルの名前。
    """

    def __init__(self, model_name: str):
        """ModelNotFoundError を初期化します。

        Args:
            model_name: 見つからなかったモデルの名前。
        """
        self.model_name = model_name
        message = f"モデル '{model_name}' が見つかりません。"
        super().__init__(message)

    def __str__(self) -> str:
        return f"モデル未検出エラー: {self.model_name}"


class OutOfMemoryError(AnnotatorError):
    """主に CUDA デバイスのメモリが不足した場合の例外。

    モデルのロード時または推論実行中に発生する可能性があります。

    Attributes:
        message: エラーの詳細メッセージ (通常は発生源の情報を含む)。
    """

    def __init__(self, message: str):
        """OutOfMemoryError を初期化します。

        Args:
            message: エラーの詳細メッセージ。
        """
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return f"メモリ不足エラー: {self.message}"


class PHashCalculationError(Exception):
    """pHash計算時のエラーを表す例外"""

    pass


class InvalidInputError(AnnotatorError):
    """無効な入力データが提供された場合の例外。

    Attributes:
        message: エラーの詳細メッセージ。
    """

    def __init__(self, message: str):
        """InvalidInputError を初期化します。

        Args:
            message: エラーの詳細メッセージ。
        """
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return f"無効な入力エラー: {self.message}"


class InvalidModelConfigError(AnnotatorError):
    """モデル設定が無効な場合の例外。

    Attributes:
        message: エラーの詳細メッセージ。
    """

    def __init__(self, message: str):
        """InvalidModelConfigError を初期化します。

        Args:
            message: エラーの詳細メッセージ。
        """
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return f"無効なモデル設定エラー: {self.message}"


class InvalidOutputError(AnnotatorError):
    """モデルの出力が無効な場合の例外。

    Attributes:
        message: エラーの詳細メッセージ。
    """

    def __init__(self, message: str):
        """InvalidOutputError を初期化します。

        Args:
            message: エラーの詳細メッセージ。
        """
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return f"無効な出力エラー: {self.message}"


class ModelExecutionError(AnnotatorError):
    """モデルの実行中にエラーが発生した場合の例外。

    Attributes:
        message: エラーの詳細メッセージ。
    """

    def __init__(self, message: str):
        """ModelExecutionError を初期化します。

        Args:
            message: エラーの詳細メッセージ。
        """
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return f"モデル実行エラー: {self.message}"


class UnsupportedModelError(AnnotatorError):
    """サポートされていないモデルが要求された場合の例外。

    Attributes:
        message: エラーの詳細メッセージ。
    """

    def __init__(self, message: str):
        """UnsupportedModelError を初期化します。

        Args:
            message: エラーの詳細メッセージ。
        """
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return f"サポートされていないモデルエラー: {self.message}"


# 必要に応じて他の特定の例外をここに追加
# 例:
# class PreprocessingError(AnnotatorError):
#     """画像の前処理中にエラーが発生した場合の例外。"""
#     pass
#
# class InferenceError(AnnotatorError):
#     """モデルの推論実行中にエラーが発生した場合の例外。"""
#     pass
