"""テスト全体で共有されるfixtures。

このモジュールでは、複数のテストファイルで使用される共通のfixtureを定義します。
"""

from unittest.mock import patch

import pytest


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
    with patch("image_annotator_lib.core.utils.load_model_config") as mock_load_config:
        mock_load_config.return_value = data
        yield data
