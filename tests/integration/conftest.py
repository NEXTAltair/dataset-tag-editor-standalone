import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# モデルディレクトリの設定
TEST_MODEL_DIR = Path(__file__).parent.parent / "resources" / "models"
TEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# devices モジュールをモック化
devices_mock = MagicMock()
devices_mock.device = torch.device("cpu")
devices_mock.torch_gc = MagicMock()
sys.modules["scripts.devices"] = devices_mock

# settings モジュールの設定
import settings
settings.current = settings.Settings(
    interrogator_model_dir=str(TEST_MODEL_DIR),
    tagger_use_spaces=False
)

@pytest.fixture(scope="session")
def test_model_dir():
    """テスト用のモデルディレクトリを提供するfixture"""
    return TEST_MODEL_DIR

@pytest.fixture(scope="session")
def devices():
    """デバイスモックを提供するfixture"""
    return devices_mock