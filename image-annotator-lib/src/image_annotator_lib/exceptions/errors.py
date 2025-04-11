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


class ConfigurationError(AnnotatorError):
    """設定に関連するエラーが発生した場合の例外。

    設定ファイルの構文エラー、必須キーの欠損、不正な値などが原因で発生します。

    Attributes:
        message: エラーの詳細メッセージ。
    """

    def __init__(self, message: str):
        """ConfigurationError を初期化します。

        Args:
            message: エラーの詳細メッセージ。
        """
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return f"設定エラー: {self.message}"


# 必要に応じて他の特定の例外をここに追加
# 例:
# class PreprocessingError(AnnotatorError):
#     """画像の前処理中にエラーが発生した場合の例外。"""
#     pass
#
# class InferenceError(AnnotatorError):
#     """モデルの推論実行中にエラーが発生した場合の例外。"""
#     pass


# Web API 関連の例外クラス
class WebApiError(AnnotatorError):
    """Web API 関連の基底例外クラス。

    Attributes:
        message: エラーの詳細メッセージ。
        provider_name: APIプロバイダー名。
    """

    def __init__(self, message: str, provider_name: str = ""):
        """WebApiError を初期化します。

        Args:
            message: エラーの詳細メッセージ。
            provider_name: APIプロバイダー名(例: "Google", "OpenAI")。
        """
        self.provider_name = provider_name
        provider_prefix = f"{provider_name} " if provider_name else ""
        self.message = f"{provider_prefix}API エラー: {message}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class ApiAuthenticationError(WebApiError):
    """API 認証に失敗した場合の例外。

    通常はAPIキーが無効、期限切れ、または権限不足の場合に発生します。

    Attributes:
        status_code: HTTP ステータスコード(通常は 401)。
    """

    def __init__(
        self,
        message: str = "API認証に失敗しました。APIキーの有効性を確認してください。",
        provider_name: str = "",
        status_code: int = 401,
    ):
        """ApiAuthenticationError を初期化します。"""
        self.status_code = status_code
        super().__init__(message, provider_name)


class ApiRateLimitError(WebApiError):
    """API レート制限に達した場合の例外。

    短時間に多くのリクエストを送信した場合に発生します。

    Attributes:
        status_code: HTTP ステータスコード(通常は 429)。
        retry_after: 再試行までの推奨待機時間(秒)。
    """

    def __init__(
        self,
        message: str = "APIのレート制限に達しました。しばらく待ってから再試行してください。",
        provider_name: str = "",
        status_code: int = 429,
        retry_after: int = 60,
    ):
        """ApiRateLimitError を初期化します。"""
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(message, provider_name)


class ApiRequestError(WebApiError):
    """API リクエストの形式または内容に問題があった場合の例外。

    通常はクライアント側のエラーです。

    Attributes:
        status_code: HTTP ステータスコード(通常は 400)。
    """

    def __init__(self, message: str, provider_name: str = "", status_code: int = 400):
        """ApiRequestError を初期化します。"""
        self.status_code = status_code
        super().__init__(f"リクエストの形式または内容に問題がありました: {message}", provider_name)


class ApiServerError(WebApiError):
    """API サーバー側でエラーが発生した場合の例外。

    サーバーの問題やメンテナンスなどで発生します。

    Attributes:
        status_code: HTTP ステータスコード(通常は 500番台)。
    """

    def __init__(self, message: str, provider_name: str = "", status_code: int = 500):
        """ApiServerError を初期化します。"""
        self.status_code = status_code
        super().__init__(f"APIサーバーエラーが発生しました: {message}", provider_name)


class ApiTimeoutError(WebApiError):
    """API リクエストがタイムアウトした場合の例外。

    ネットワーク問題や負荷の高いリクエストで発生します。
    """

    def __init__(
        self,
        message: str = "APIリクエストがタイムアウトしました。ネットワーク状況やAPIの状態を確認してください。",
        provider_name: str = "",
    ):
        """ApiTimeoutError を初期化します。"""
        super().__init__(message, provider_name)


class ApiKeyMissingError(WebApiError):
    """API キーが環境変数に設定されていない場合の例外。"""

    def __init__(self, env_var: str, provider_name: str = ""):
        """ApiKeyMissingError を初期化します。

        Args:
            env_var: APIキーが格納されるべき環境変数名。
            provider_name: APIプロバイダー名。
        """
        self.env_var = env_var
        message = f"APIキーが環境変数 {env_var} に設定されていません。"
        super().__init__(message, provider_name)


class InsufficientCreditsError(WebApiError):
    """APIのクレジット残高が不足している場合の例外。

    Attributes:
        status_code: HTTP ステータスコード(通常は 402)。
    """

    def __init__(
        self,
        message: str = "APIのクレジット残高が不足しています。アカウントを確認してください。",
        provider_name: str = "",
        status_code: int = 402,
    ):
        """InsufficientCreditsError を初期化します。"""
        self.status_code = status_code
        super().__init__(message, provider_name)
