"""テスト全体で共有されるfixtures。

このモジュールでは、複数のテストファイルで使用される共通のfixtureを定義します。
"""

from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from PIL import Image


# 警告を無視するための設定
def pytest_configure(config: pytest.Config) -> None:
    """pytestの設定を構成する"""
    # 特定の警告を無視
    config.addinivalue_line("filterwarnings", "ignore::FutureWarning:transformers.*")


# resourcesディレクトリのパス
resources_dir = Path("tests") / "resources"


@pytest.fixture(scope="session")
def load_image_files() -> Callable[[int], list[Image.Image]]:
    """指定された枚数の画像ファイルをリストとして読み込む関数を返す"""

    def _load_images(count: int = 1) -> list[Image.Image]:
        image_path = resources_dir / "img" / "1_img"
        files = list(image_path.rglob("*.webp"))

        # 指定された枚数だけファイルを取得(ディレクトリ内のファイル数を超えないように)
        count = min(count, len(files))
        files = files[:count]

        # すべての画像をリストに格納して返す
        return [Image.open(file) for file in files if file.exists()]

    return _load_images


@pytest.fixture
def mock_test_config_toml() -> Generator[dict[str, Any], None, None]:
    data = {
        "test_model_01": {
            "model_path": " path/to/test_model_01",
            "device": "cuda",
            "score_prefix": "[TEST01]",
            "class": "TestScorer01",
        },
        "test_model_02": {
            "model_path": "path/to/test_model_02",
            "score_prefix": "[TEST02]",
            "class": "TestScorer02",
        },
    }
    with patch("image_annotator_lib.core.utils.load_model_config") as mock_load_config:
        mock_load_config.return_value = data
        yield data
