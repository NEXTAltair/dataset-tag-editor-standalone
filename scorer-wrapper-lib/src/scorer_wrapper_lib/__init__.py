from .core.utils import setup_logger
from .scorer import evaluate
from .scorer_registry import list_available_scorers

# アプリケーション全体のロガー設定
setup_logger("scorer_wrapper_lib")

__all__ = ["evaluate", "list_available_scorers"]

# list_available_scorers() で使えるモデルリストを返す
# evaluate(images_lis, model_name_list) で評価を実行する
# 使える機能はコレだけ
