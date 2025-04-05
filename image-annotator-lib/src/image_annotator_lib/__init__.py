import logging
import os
from typing import Any

from PIL import Image

# NOTE: TensorFlowの表示抑制用､見ててうざいだろ
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from .core.base import AnnotationResult
from .core.utils import setup_logger
from .exceptions.errors import (
    AnnotatorError,
    ModelLoadError,
    ModelNotFoundError,
    OutOfMemoryError,
)

setup_logger("image_annotator_lib", level=logging.DEBUG)


# --- Public API ---

__all__ = [
    "AnnotationResult",
    "AnnotatorError",
    "ModelLoadError",
    "ModelNotFoundError",
    "OutOfMemoryError",
    "annotate",
    "list_available_annotators",
]

# モジュールレベルのキャッシュ
# NOTE: 遅延インポートにして必要なときだけregistryとannotatorをインポートしないと他のテストが激遅になる
_cached_list_available_annotators = None
_cached_annotate = None


def list_available_annotators() -> list[str]:
    """利用可能なアノテーターモデルのリストを返します。"""
    global _cached_list_available_annotators
    if _cached_list_available_annotators is None:
        # pylint: disable=import-outside-toplevel
        from .core.registry import list_available_annotators as _list_available_annotators_impl

        _cached_list_available_annotators = _list_available_annotators_impl
    return _cached_list_available_annotators()


def annotate(
    images_list: list[Image.Image], model_name_list: list[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    指定されたモデルを使用して画像のリストにアノテーションを付けます。

    Args:
        images_list: アノテーションを付けるPIL Imageオブジェクトのリスト。
        model_name_list: 使用するアノテーターモデル名のリスト。

    Returns:
        モデル名をキーとし、各画像のアノテーション結果(AnnotationResult)のリストを
        値とする辞書。
    """
    global _cached_annotate
    if _cached_annotate is None:
        # pylint: disable=import-outside-toplevel
        from .api import annotate as _annotate_impl

        _cached_annotate = _annotate_impl
    # api.pyのannotate関数に合わせて引数名を変更 (model_name_list -> model_names)
    return _cached_annotate(images=images_list, model_names=model_name_list)


# You might want to add version information here later
# __version__ = "0.1.0"
