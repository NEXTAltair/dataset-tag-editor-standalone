from typing import Any

from PIL import Image

from .core.utils import setup_logger

# アプリケーション全体のロガー設定
setup_logger("scorer_wrapper_lib")

__all__ = ["evaluate", "list_available_models"]


# モジュールレベルのキャッシュ
# NOTE: 遅延インポートにして必要なときだけscorer_registryとscorerをインポートしないと他のテストが激遅になる
_cached_list_available_models = None
_cached_evaluate = None


def list_available_models() -> list[str]:
    global _cached_list_available_models
    if _cached_list_available_models is None:
        from .scorer_registry import list_available_scorers as _list_available_scorers

        _cached_list_available_models = _list_available_scorers
    return _list_available_scorers()


def evaluate(images_list: list[Image.Image], model_name_list: list[str]) -> dict[str, list[dict[str, Any]]]:
    global _cached_evaluate
    if _cached_evaluate is None:
        from .scorer import evaluate as _evaluate

        _cached_evaluate = _evaluate
    return _cached_evaluate(images_list, model_name_list)


# list_available_scorers() で使えるモデルリストを返す
# evaluate(images_list, model_name_list) で評価を実行する
# 使える機能はコレだけ
