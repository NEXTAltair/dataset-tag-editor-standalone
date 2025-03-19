import logging
from typing import Any

from PIL import Image

from .core.utils import setup_logger

# アプリケーション全体のロガー設定
setup_logger("tagger_wrapper_lib", level=logging.INFO)

__all__ = ["evaluate", "list_available_models"]


# モジュールレベルのキャッシュ
# NOTE: 遅延インポートにして必要なときだけregistryとtaggerをインポートしないと他のテストが激遅になる
_cached_list_available_models = None
_cached_evaluate = None


def list_available_models() -> list[str]:
    global _cached_list_available_models
    if _cached_list_available_models is None:
        from .registry import list_available_taggers as _list_available_taggers

        _cached_list_available_models = _list_available_taggers
    return _list_available_taggers()


def evaluate(images_list: list[Image.Image], model_name_list: list[str]) -> dict[str, list[dict[str, Any]]]:
    global _cached_evaluate
    if _cached_evaluate is None:
        from .tagger import evaluate as _evaluate

        _cached_evaluate = _evaluate
    return _cached_evaluate(images_list, model_name_list)


# list_available_taggers() で使えるモデルリストを返す
# evaluate(images_list, model_name_list) でアノテーションを実行する
# 使える機能はコレだけ
