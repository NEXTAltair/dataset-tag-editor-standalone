import logging
from typing import Any

from PIL import Image

from .scorer_registry import get_cls_obj_registry

logger = logging.getLogger(__name__)

_MODEL_INSTANCE_REGISTRY: dict[str, Any] = {}


def _create_scorer_instance(model_name: str) -> Any:
    """
    _MODEL_INSTANCE_REGISTRYに登録されているモデルに対応したクラスを取得し、
    モデル名を引数にモデルインスタンスを生成

    Args:
        model_name (str): モデルの名前。

    Returns:
        BaseScorer: スコアラーのインスタンス。
    """
    registry = get_cls_obj_registry()
    scorer_class = registry[model_name]
    instance = scorer_class(model_name=model_name)
    logger.debug(
        f"モデル '{model_name}' の新しいインスタンスを作成しました (クラス: {scorer_class.__name__})"
    )
    return instance


def get_scorer_instance(model_name: str) -> Any:
    """モデル名からスコアラーインスタンスを取得する

    モデルがすでにロードされている場合はキャッシュから返す。
    まだロードされていない場合は、新たにインスタンスを作成してキャッシュに保存する。

    Args:
        model_name: モデルの名前（models.tomlで定義されたキー）

    Returns:
        スコアラーインスタンス

    Raises:
        ValueError: 指定されたモデル名が設定に存在しない場合
    """
    if model_name in _MODEL_INSTANCE_REGISTRY:
        logger.debug(f"モデル '{model_name}' はキャッシュから取得されました")
        return _MODEL_INSTANCE_REGISTRY[model_name]

    instance = _create_scorer_instance(model_name)
    _MODEL_INSTANCE_REGISTRY[model_name] = instance
    return instance


def _evaluate_model(scorer: Any, images: list[Image.Image]) -> list[dict[str, Any]]:
    """1モデル分の評価処理を実施します。
    ・モデルのロード／復元、予測、キャッシュ＆リリースを実行
    """
    scorer.load_or_restore_model()  # ロードまたは復元
    results: list[dict[str, Any]] = scorer.predict(images)  # 予測結果を取得
    logger.debug(f"モデル '{scorer.model_name}' の評価結果を統一した形式に変換結果: {results}")
    scorer.cache_to_main_memory()
    return results


def evaluate(images: list[Image.Image], model_list: list[str]) -> dict[str, list[dict[str, Any]]]:
    """画像評価の処理を実施します。
       ・各モデルに対して評価を実行し、最終的に結果を集約・グループ化します。

    Args:
        images: 評価対象の画像リスト
        results_by_model: 使用するモデルのリスト

    Returns:
        モデルごとにグループ化した結果

    戻り値の例:
        {
            "aesthetic_shadow_v1": [
                {"model_output": ..., "model_name": "aesthetic_shadow_v1", "score_tag": "aesthetic"},
                {"model_output": ..., "model_name": "aesthetic_shadow_v1", "score_tag": "aesthetic"}
            ],
            "ImprovedAesthetic": [
                {"model_output": ..., "model_name": "ImprovedAesthetic", "score_tag": "[IAP]score_8"},
                {"model_output": ..., "model_name": "ImprovedAesthetic", "score_tag": "[IAP]score_4"}
            ]
        }
        ※各リストの要素は画像ごとの評価結果（辞書）で、リストの長さは入力画像数と一致
    """
    logger.info(f"{len(images)}枚の画像を{len(model_list)}個のモデルで評価します: {model_list}")
    results_by_model: dict[str, list[dict[str, Any]]] = {}

    for model_name in model_list:
        logger.info(f"モデル '{model_name}' での評価を開始します")
        scorer = get_scorer_instance(model_name)
        results = _evaluate_model(scorer, images)
        # 結果をモデルごとに集約
        for result in results:
            results_by_model.setdefault(model_name, []).append(result)
        logger.info(f"モデル '{model_name}' の評価が完了しました")

    return results_by_model


def release_resources() -> None:
    """
    モデルの握ったリソースを解放。
    """
    for scorer in _MODEL_INSTANCE_REGISTRY.values():
        scorer.release_resources()
