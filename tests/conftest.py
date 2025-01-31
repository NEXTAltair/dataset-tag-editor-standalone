import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# ルートディレクトリの設定
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# グローバルモックの作成
devices_mock = MagicMock(spec=True)
devices_mock.device = torch.device("cpu")
devices_mock.cpu = torch.device("cpu")

cmd_args_mock = MagicMock()
logger_mock = MagicMock()


@pytest.fixture(scope="session", autouse=True)
def setup_global_mocks():
    """グローバルなモックの設定"""
    mocks = {
        "cmd_args": MagicMock(),
        "logger": MagicMock(),
        "devices": MagicMock(
            spec=True, device=torch.device("cpu"), cpu=torch.device("cpu")
        ),
    }

    with patch.dict("sys.modules", mocks):
        yield mocks

@pytest.fixture
def mock_logger(setup_global_mocks):
    return setup_global_mocks["logger"]

def pytest_collection_modifyitems(items):
    items.sort(key=lambda x: x.get_closest_marker("dependency", None) is not None)


def pytest_configure(config):
    """pytestの設定をカスタマイズ"""
    sys.argv = [sys.argv[0]]  # __main__.pyの引数を無視
