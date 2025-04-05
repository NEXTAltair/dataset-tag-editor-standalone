# Stub file for image_annotator_lib.exceptions.errors

class AnnotatorError(Exception):
    """Base exception for the image-annotator-lib. / image-annotator-lib の基底例外クラス。"""

    ...

class ModelLoadError(AnnotatorError):
    """Exception raised for errors during model loading. / モデルのロード中に発生したエラー。

    Attributes:
        message (str): Human readable string describing the error. / エラー内容を示す文字列。
    """

    message: str
    def __init__(self, message: str) -> None: ...

class ModelNotFoundError(AnnotatorError):
    """Exception raised when a requested model is not found. / 要求されたモデルが見つからない場合に発生するエラー。

    Attributes:
        model_name (str): The name of the model that was not found. / 見つからなかったモデルの名前。
    """

    model_name: str
    def __init__(self, model_name: str) -> None: ...

class OutOfMemoryError(AnnotatorError):
    """Exception raised when CUDA runs out of memory. / CUDA メモリが不足した場合に発生するエラー。

    Attributes:
        message (str): Human readable string describing the error. / エラー内容を示す文字列。
    """

    message: str
    def __init__(self, message: str) -> None: ...

# Add other specific exceptions as needed / 必要に応じて他の特定の例外を追加
