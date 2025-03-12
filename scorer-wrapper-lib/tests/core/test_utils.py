from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from scorer_wrapper_lib.core.utils import get_model_config, load_image_from_path, validate_model_config


@pytest.fixture
def mock_image_path(tmp_path):
    """テスト用の一時画像ファイルを作成"""
    img_path = tmp_path / "test_image.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)
    return str(img_path)


def test_load_image_from_path(mock_image_path):
    """画像読み込み関数のテスト"""
    image = load_image_from_path(mock_image_path)
    assert isinstance(image, Image.Image)
    assert image.size == (100, 100)


def test_load_image_invalid_path():
    """無効なパスからの画像読み込みテスト"""
    with pytest.raises(Exception):
        load_image_from_path("invalid/path/to/image.jpg")


def test_get_model_config():
    """モデル設定取得のテスト"""
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", MagicMock()):
            with patch("json.load", return_value={"model_type": "test"}):
                config = get_model_config("test_model")
                assert config["model_type"] == "test"


def test_validate_model_config():
    """モデル設定検証のテスト"""
    valid_config = {"model_type": "test", "model_path": "/path/to/model", "required_field": "value"}

    # 必須フィールドを指定してテスト
    result = validate_model_config(valid_config, ["model_type", "model_path"])
    assert result is True

    # 無効な設定でテスト
    invalid_config = {"model_type": "test"}
    with pytest.raises(Exception):
        validate_model_config(invalid_config, ["model_path"])
