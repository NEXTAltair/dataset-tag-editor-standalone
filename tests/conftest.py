import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

# ルートディレクトリの設定
root_dir = Path(__file__).resolve().parent.parent
scripts_dir = root_dir / "scripts"
sys.path.extend([str(root_dir), str(scripts_dir)])

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
        "print_color": MagicMock(),
        "launch": MagicMock(),
        "utilities": MagicMock(),
        "settings": MagicMock(current=MagicMock(tagger_use_spaces=False)),
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

    # 必要なモジュールをモック化
    mock_modules = {
        "print_color": MagicMock(print=print),
        "cmd_args": MagicMock(),
        "logger": MagicMock(),
        "devices": MagicMock(
            spec=True, device=torch.device("cpu"), cpu=torch.device("cpu")
        ),
        "utilities": MagicMock(),
        "settings": MagicMock(current=MagicMock(tagger_use_spaces=False)),
        "launch": MagicMock(),
        "scripts.dataset_tag_editor.interrogators": MagicMock(
            BLIPLargeCaptioning=MagicMock,
            BLIP2Captioning=MagicMock,
            GITLargeCaptioning=MagicMock,
            WaifuDiffusionTagger=MagicMock,
            DepDanbooruTagger=MagicMock,
            WaifuDiffusionTaggerTimm=MagicMock,
        ),
    }

    # モジュールをsys.modulesに追加
    for name, mock in mock_modules.items():
        sys.modules[name] = mock


@pytest.fixture
def test_image():
    """テスト用の画像を提供するfixture"""
    img_path = Path(__file__).parent / "resources" / "img" / "1_img" / "file01.webp"
    return Image.open(img_path)


@pytest.fixture
def mock_interrogator():
    """interrogatorのベースモック"""
    mock = MagicMock()
    mock.load = MagicMock()
    mock.unload = MagicMock()
    mock.apply = MagicMock()
    return mock
    sys.argv = [sys.argv[0]]  # __main__.pyの引数を無視
