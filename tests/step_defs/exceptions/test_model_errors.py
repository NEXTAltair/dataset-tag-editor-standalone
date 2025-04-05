import pytest

from image_annotator_lib.exceptions.errors import (
    InvalidInputError,
    InvalidModelConfigError,
    InvalidOutputError,
    ModelExecutionError,
    ModelLoadError,
    ModelNotFoundError,
    UnsupportedModelError,
)


def test_model_exceptions():
    """例外クラスのテスト"""
    # 各例外クラスのインスタンス化テスト
    with pytest.raises(ModelNotFoundError):
        raise ModelNotFoundError("テストエラー")

    with pytest.raises(ModelLoadError):
        raise ModelLoadError("テストエラー")

    with pytest.raises(InvalidModelConfigError):
        raise InvalidModelConfigError("テストエラー")

    with pytest.raises(UnsupportedModelError):
        raise UnsupportedModelError("テストエラー")

    with pytest.raises(ModelExecutionError):
        raise ModelExecutionError("テストエラー")

    with pytest.raises(InvalidInputError):
        raise InvalidInputError("テストエラー")

    with pytest.raises(InvalidOutputError):
        raise InvalidOutputError("テストエラー")
