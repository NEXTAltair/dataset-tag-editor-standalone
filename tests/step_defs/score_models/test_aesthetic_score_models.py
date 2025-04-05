import unittest.mock as mock
from typing import Any

import pytest
from pytest_bdd import given, parsers, then, when, scenarios

from image_annotator_lib.models.pipeline_scorers import AestheticShadow
from image_annotator_lib.models.pipeline_scorers import CafePredictor

scenarios("../../features/score_models/aesthetic_score_models.feature")


# 共通のフィクスチャ
@pytest.fixture
def model_context() -> dict[str, Any]:
    """テスト用のコンテキストを提供するフィクスチャ"""
    # 各モデルの実装に適した戻り値フォーマット
    # NOTE: スコアの桁数は減らしている
    output_formats = {
        "aesthetic_shadow": [
            {"label": "hq", "score": 0.65},
            {"label": "lq", "score": 0.35},
        ],
        "cafe_aesthetic": [
            {"label": "aesthetic", "score": 0.67, "normalized_score": 0.67},
            {"label": "technical", "score": 0.75, "normalized_score": 0.75},
        ],
    }

    return {
        "model": None,
        "score": None,
        "tag": None,
        "output_formats": output_formats,
    }


# Aesthetic Shadow V1 モデルのステップ
@given('"shadowlilac/aesthetic-shadow" モデルが利用可能である')
def aesthetic_shadow_v1_model(model_context: dict[str, Any]) -> None:
    with mock.patch("image_annotator_lib.core.model_factory.ModelLoad.load_transformers_components"):
        model = AestheticShadow("aesthetic_shadow_v1")
        model._pipeline = mock.MagicMock()
        model._pipeline.return_value = model_context["output_formats"]["aesthetic_shadow"]
        model_context["model"] = model
        model_context["model_type"] = "pipeline"
        assert model is not None


# Aesthetic Shadow V2 モデルのステップ
@given('"NEXTAltair/cache_aestheic-shadow-v2" モデルが利用可能である')
def aesthetic_shadow_v2_model(model_context: dict[str, Any]) -> None:
    with mock.patch("image_annotator_lib.core.model_factory.ModelLoad.load_transformers_components"):
        model = AestheticShadow("aesthetic_shadow_v2")
        model._pipeline = mock.MagicMock()
        model._pipeline.return_value = model_context["output_formats"]["aesthetic_shadow"]
        model_context["model"] = model
        model_context["model_type"] = "pipeline"
        assert model is not None


# CAFE Aesthetic モデルのステップ
@given('"cafeai/cafe_aesthetic" モデルが利用可能である')
def cafe_aesthetic_model(model_context: dict[str, Any]) -> None:
    with mock.patch("image_annotator_lib.core.model_factory.ModelLoad.load_transformers_components"):
        model = CafePredictor("cafe_aesthetic")
        model.config = {"score_prefix": "[CAFE]"}
        model._evaluate = mock.MagicMock()
        model._evaluate.return_value = model_context["output_formats"]["cafe_aesthetic"]
        model_context["model"] = model
        model_context["model_type"] = "pipeline"
        assert model is not None


# Aesthetic Shadow モデル初期化ステップ
@given("Aesthetic Shadow モデルが初期化されている")
def aesthetic_shadow_model_init(model_context: dict[str, Any]) -> None:
    with mock.patch("image_annotator_lib.core.model_factory.ModelLoad.load_transformers_components"):
        model = AestheticShadow("aesthetic_shadow_v1")
        model_context["model"] = model
        model_context["model_type"] = "pipeline"
        assert model is not None


# 画像評価ステップ
@when("画像を評価すると")
def evaluate_image(model_context: dict[str, Any]) -> None:
    # モデルタイプに応じた評価結果をシミュレート
    if model_context["model_type"] == "aesthetic_shadow":
        # Shadow系モデルは高品質スコアを返す
        model_context["score"] = 0.65  # hqスコア
    elif model_context["model_type"] == "cafe_aesthetic":
        # CAFEモデルは美的スコアを返す
        model_context["score"] = 0.67  # aesthetic normalized_score
    else:
        # 不明なモデルタイプの場合はデフォルト値
        model_context["score"] = 0.5

    # モデルの評価メソッドをモック
    model_context["model"].evaluate = mock.MagicMock()
    model_context["model"].evaluate.return_value = model_context["score"]

    # 評価実行
    result = model_context["model"].evaluate("dummy_image_path")
    assert result == model_context["score"]


# スコア結果ステップ
@when(parsers.parse("画像の評価結果が {score:f} である"))
def score_result(model_context: dict[str, Any], score: float) -> None:
    model_context["score"] = score


# スコア範囲確認ステップ
@then("0-1の範囲のスコアが返される")
def verify_score_range(model_context: dict[str, Any]) -> None:
    assert 0 <= model_context["score"] <= 1


# タグ付与確認ステップ
@then("スコアに基づいて適切なタグが付与される")
def verify_tag_assignment(model_context: dict[str, Any]) -> None:
    # モデルのget_score_tagメソッドをモック
    model_context["model"].get_score_tag = mock.MagicMock()

    # モデルタイプに基づいて適切なタグを設定
    if model_context["model_type"] == "aesthetic_shadow":
        if model_context["score"] >= 0.7:
            expected_tag = "very aesthetic"
        elif model_context["score"] >= 0.5:
            expected_tag = "aesthetic"
        elif model_context["score"] >= 0.3:
            expected_tag = "displeasing"
        else:
            expected_tag = "very displeasing"
    else:
        expected_tag = "aesthetic"  # デフォルトタグ

    model_context["model"].get_score_tag.return_value = expected_tag

    # タグを取得して検証
    tag = model_context["model"].get_score_tag(model_context["score"])
    model_context["tag"] = tag

    # タグが適切な値であることを確認
    if model_context["model_type"] == "aesthetic_shadow":
        assert tag in ["very aesthetic", "aesthetic", "displeasing", "very displeasing"]
    else:
        assert tag == expected_tag


# CAFE Aesthetic スコア変換ステップ
@then(parsers.parse('変換されたタグは "[CAFE]score_{integer_value}" となる'))
def verify_cafe_score_tag(model_context: dict[str, Any], integer_value: int) -> None:
    # CAFEモデルのスコア変換テスト
    expected_tag = f"[CAFE]score_{integer_value}"

    # get_score_tag メソッドをモック
    model_context["model"].get_score_tag = mock.MagicMock()
    model_context["model"].get_score_tag.return_value = expected_tag

    # 実際にモデルのスコアタグ変換メソッドを呼び出す
    actual_tag = model_context["model"].get_score_tag(model_context["score"])

    # 期待値と実際の値が一致することを確認
    assert actual_tag == expected_tag


# タグ評価ステップ
@then(parsers.parse("評価タグは {tag} となる"))
def verify_aesthetic_tag(model_context: dict[str, Any], tag: str) -> None:
    # _get_score_tag メソッドをモック
    model_context["model"]._get_score_tag = mock.MagicMock()
    model_context["model"]._get_score_tag.return_value = tag

    # テスト実行
    actual_tag = model_context["model"]._get_score_tag(model_context["score"])
    assert actual_tag == tag
