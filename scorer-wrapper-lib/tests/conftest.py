"""テスト全体で共有されるfixtures。

このモジュールでは、複数のテストファイルで使用される共通のfixtureを定義します。
"""

import sys
import tempfile
from pathlib import Path

from PIL import Image
from pytest_bdd import given

from scorer_wrapper_lib.core.utils import ConsoleLogCapture

# ルートパスの設定
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))


# Given ステップ
@given("有効な画像が用意されている", target_fixture="single_image")
def given_single_image():
    return [Image.open("tests/resources/img/1_img/file01.webp")]


@given("複数の有効な画像が用意されている", target_fixture="images")
def given_image_list():
    return [
        Image.open("tests/resources/img/1_img/file01.webp"),
        Image.open("tests/resources/img/1_img/file02.webp"),
    ]


@given("コンソールログキャプチャツールが初期化される")
def init_console_log_capture():
    """コンソールログキャプチャツールを初期化するステップ。"""
    return ConsoleLogCapture()


@given("ログファイルパスが指定される")
def specify_log_file_path():
    """ログファイルパスを指定するステップ。"""
    temp_dir = tempfile.gettempdir()
    log_file = Path(temp_dir) / "console_log_test.log"
    # テスト前にファイルが存在していたら削除
    if log_file.exists():
        log_file.unlink(missing_ok=True)
    return log_file


@given("標準出力のみキャプチャする設定がされる")
def init_stdout_only_capture():
    """標準出力のみキャプチャする設定を行うステップ。"""
    return ConsoleLogCapture(capture_stderr=False)
