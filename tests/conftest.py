"""テスト全体で共有されるfixtures。

このモジュールでは、複数のテストファイルで使用される共通のfixtureを定義します。
"""

from unittest.mock import patch

import pytest
from PIL import Image
from pytest_bdd import given, parsers


# 警告を無視するための設定
def pytest_configure(config):
    """pytestの設定を構成する"""
    # 特定の警告を無視
    config.addinivalue_line("filterwarnings", "ignore::FutureWarning:transformers.*")


@pytest.fixture
def mock_config_toml():
    data = {
        "test_model_01": {
            "type": "pipeline",
            "model_path": " path/to/test_model_01",
            "device": "cuda",
            "score_prefix": "[TEST01]",
            "class": "TestScorer01",
        },
        "test_model_02": {
            "type": "ClipClassifierModel",
            "model_path": "path/to/test_model_02",
            "device": "cpu",
            "score_prefix": "[TEST02]",
            "class": "TestScorer02",
        },
    }
    with patch("scorer_wrapper_lib.core.utils.load_model_config") as mock_load_config:
        mock_load_config.return_value = data
        yield data


# スコアラーテスト用のGivenステップ
@given("単一の画像を用意する", target_fixture="target_images")
def use_single_image(single_image: list[Image.Image]) -> list[Image.Image]:
    """single_imageフィクスチャを流用するステップ"""
    return single_image


@given("複数の画像を用意する", target_fixture="target_images")
def use_multiple_images(images: list[Image.Image]) -> list[Image.Image]:
    """imagesフィクスチャを流用するステップ"""
    return images


@given(
    parsers.parse("モデル {model_spec} が指定される"),
    target_fixture="target_model_list",
)
def specify_models(model_spec: str) -> list[str]:
    """モデル指定を処理するステップ（単一・複数両対応）

    Args:
        model_spec: モデル指定文字列。以下の形式をサポート:
            - 単一モデル名: "ModelName"
            - カンマ区切りリスト: "Model1, Model2"
            - JSON形式リスト: '["Model1", "Model2"]'
    """
    # 最初にクォートを除去
    clean_spec = model_spec.strip("\"'")

    # カンマ区切りの場合
    if "," in clean_spec and not clean_spec.startswith("["):
        return [name.strip() for name in clean_spec.split(",")]

    # JSON形式リストの場合
    if clean_spec.startswith("["):
        try:
            import ast

            result = ast.literal_eval(clean_spec)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            pass

    # 単一モデル名の場合
    return [clean_spec]
