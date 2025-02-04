import sys

import pytest

# from scripts import devices  # devices.py はインポートしない (問題の原因)
from tests.integration import devices_test  # devices_test.py を使用 (統合テスト用)


@pytest.fixture
def cpu_device():
    """CPUデバイスを提供するfixture (devices_test.py 使用)"""
    return devices_test.cpu  # devices_test.cpu をそのまま返す


@pytest.fixture
def cuda_device():
    """CUDAデバイスを提供するfixture (devices_test.py 使用)"""
    return devices_test.get_cuda_device()  # devices_test.get_cuda_device() をそのまま返す


@pytest.fixture(scope="session", autouse=True)
def disable_mocks():
    """統合テスト用に既存のモックを無効化するfixture

    統合テストでは実際のモジュールを使用するため、
    tests.conftest.pyで設定されたモックを上書きする
    """
    # 実際のモジュールをインポート
    import cmd_args
    import launch
    import logger
    import settings
    import utilities

    from scripts.dataset_tag_editor import interrogators

    # モックを実際のモジュールで上書き
    mocks = {
        "cmd_args": cmd_args,
        "logger": logger,
        "utilities": utilities,
        "settings": settings,
        "launch": launch,
        "scripts.dataset_tag_editor.interrogators": interrogators,
    }

    with pytest.MonkeyPatch.context() as mp:
        for name, module in mocks.items():
            mp.setitem(sys.modules, name, module)
        yield
