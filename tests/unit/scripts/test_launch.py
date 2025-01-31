import subprocess
from collections import namedtuple
from unittest.mock import MagicMock, patch

import pytest
import torch

# グローバル変数としてlaunchを定義
launch = None


@pytest.fixture(scope="module", autouse=True)
def setup_launch(setup_global_mocks):
    """setup_global_mocksを使用してlaunchモジュールをインポート"""
    global launch
    # devicesモックを作成
    devices_mock = MagicMock()
    devices_mock.device = torch.device("cpu")

    with patch.dict("sys.modules", {"devices": devices_mock}):
        import scripts.launch

        launch = scripts.launch
        return launch


@pytest.fixture
def mock_subprocess_run():
    with patch("subprocess.run") as mock:
        yield mock


@pytest.fixture
def mock_importlib():
    with patch("importlib.util.find_spec") as mock:
        yield mock


def test_check_python_version_compatible(mock_logger):
    version = namedtuple("Version", ["major", "minor", "micro"])
    with patch("sys.version_info", version(3, 9, 0)):
        launch.check_python_version()
        mock_logger.error.assert_not_called()


def test_check_python_version_incompatible(mock_logger):
    version = namedtuple("Version", ["major", "minor", "micro"])
    with patch("sys.version_info", version(3, 8, 0)):
        launch.check_python_version()
        mock_logger.error.assert_called_once()


@pytest.mark.parametrize(
    "package,exists", [("existing_package", True), ("non_existing_package", False)]
)
def test_is_installed(mock_importlib, package, exists):
    if exists:
        mock_importlib.return_value = MagicMock()
    else:
        mock_importlib.side_effect = ModuleNotFoundError()

    assert launch.is_installed(package) == exists


def test_run(mock_subprocess_run):
    command = "test command"
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args=command, returncode=0, stdout=b"", stderr=b""
    )

    result = launch.run(command)

    mock_subprocess_run.assert_called_once_with(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        env=launch.os.environ,
    )
    assert result.returncode == 0


def test_run_pip_success(mock_subprocess_run):
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args="pip command", returncode=0, stdout=b"", stderr=b""
    )

    launch.run_pip("install package", "test package")
    mock_subprocess_run.assert_called_once()


def test_run_pip_failure(mock_subprocess_run):
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args="pip command", returncode=1, stdout=b"error", stderr=b"error"
    )

    with pytest.raises(RuntimeError):
        launch.run_pip("install package", "test package")


@patch("scripts.launch.is_installed")
@patch("subprocess.run")
@patch("scripts.launch.cmd_args.opts")
def test_prepare_environment_no_torch(
    mock_opts, mock_subprocess_run, mock_is_installed
):
    """torchがインストール済みでforce_install_torchがNoneの場合、
    subprocess.runは呼ばれないことを確認するテスト"""
    # force_install_torch = None かつ is_installed = True の場合、
    # torchのインストールは実行されない
    mock_opts.force_install_torch = None
    mock_is_installed.return_value = True
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args="", returncode=0, stdout=b"", stderr=b""
    )

    # devicesモックを作成
    devices_mock = MagicMock()
    devices_mock.device = torch.device("cpu")

    with patch.dict("sys.modules", {"devices": devices_mock}):
        launch.prepare_environment()

    # subprocess.runが呼ばれないことを確認
    mock_subprocess_run.assert_not_called()


@patch("scripts.launch.is_installed")
@patch("subprocess.run")
@patch("scripts.launch.cmd_args.opts")
def test_prepare_environment_force_cpu(
    mock_opts, mock_subprocess_run, mock_is_installed
):
    """force_install_torch='cpu'の場合、CPUバージョンのtorchを
    インストールするためにsubprocess.runが呼ばれることを確認するテスト"""
    # force_install_torch = 'cpu' の場合、
    # is_installedの結果に関わらずtorchのインストールが実行される
    mock_opts.force_install_torch = "cpu"
    mock_is_installed.return_value = False
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args="", returncode=0, stdout=b"", stderr=b""
    )

    # devicesモックを作成
    devices_mock = MagicMock()
    devices_mock.device = torch.device("cpu")

    with patch.dict("sys.modules", {"devices": devices_mock}):
        launch.prepare_environment()

    # subprocess.runが一度呼ばれることを確認
    mock_subprocess_run.assert_called_once()
