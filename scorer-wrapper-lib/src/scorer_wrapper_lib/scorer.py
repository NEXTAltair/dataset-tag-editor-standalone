from typing import Any

from PIL import Image

from .core.utils import load_model_config
from .scorer_registry import get_registry

_LOADED_SCORERS: dict[str, Any] = {}


def _create_scorer_instance(class_name: str, model_name: str):
    """スコアラークラスのインスタンスを生成します。

    Args:
        class_name (str): スコアラークラスの名前。
        model_name (str): モデルの名前。

    Returns:
        BaseScorer: スコアラーのインスタンス。
    """
    registry = get_registry()
    scorer_class = registry[class_name]
    return scorer_class(model_name=model_name)


def init_scorer(model_name: str):
    if model_name in _LOADED_SCORERS:
        return _LOADED_SCORERS[model_name]

    model_config = load_model_config().get(model_name)
    if not model_config:
        raise ValueError(f"Model config not found for: {model_name}")

    class_name = model_config["class"]
    scorer = _create_scorer_instance(class_name, model_name)
    _LOADED_SCORERS[model_name] = scorer
    return scorer


def _evaluate_model(scorer, images: list[Image.Image]) -> list[dict]:
    """1モデル分の評価処理を実施します。
    ・モデルのロード／復元、予測、キャッシュ＆リリースを実行
    """
    scorer.load_or_restore_model()  # ロードまたは復元
    results = scorer.predict(images)  # 予測結果を取得
    scorer.cache_and_release_model()  # 終了後にリソース解放
    return results


def evaluate(images: list[Image.Image], model_list: list[str]) -> dict[str, list[dict]]:
    """画像評価の処理を実施します。
       ・各モデルに対して評価を実行し、最終的に結果を集約・グループ化します。

    Args:
        images: 評価対象の画像リスト
        model_list: 使用するモデルのリスト

    Returns:
        モデルごとにグループ化した結果
    """
    results_by_model: dict[str, list[dict]] = {}

    for name in model_list:
        scorer = init_scorer(name)  # モデルを初期化
        results = _evaluate_model(scorer, images)  # モデルを評価

        # 結果をモデルごとに集約
        for result in results:
            model_name = result["model_name"]
            results_by_model.setdefault(model_name, []).append(result)

    return results_by_model
