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


def _group_results(
    all_results: list[dict], image_list: list[Image.Image]
) -> dict[str, list[dict]] | list[dict]:
    """複数モデルの場合の評価結果をモデルごとにグループ化します。"""
    results_by_model: dict[str, list[dict]] = {}
    for result in all_results:
        model_name = result["model_name"]
        results_by_model.setdefault(model_name, []).append(result)
    if len(image_list) == 1:
        # 単一画像の場合は各モデルの最初の結果のみ返す
        return {model: [results[0]] for model, results in results_by_model.items()}  # 値をリスト型に変更
    return results_by_model


def evaluate(image_list: list[Image.Image], model_list: list[str]) -> list[dict] | dict:
    """画像評価の処理を実施します。
       ・各モデルに対して評価を実行し、最終的に結果を集約・グループ化します。
    Args:
        image_list: 評価対象の画像リスト
        model_list: 使用するモデルのリスト
    Returns:
        複数モデルならモデルごとにグループ化した結果、単一モデルならリスト形式の結果
    """
    all_results = []
    for name in model_list:
        scorer = init_scorer(name)
        results = _evaluate_model(scorer, image_list)
        all_results.extend(results)
    if len(model_list) > 1:
        return _group_results(all_results, image_list)
    return all_results
