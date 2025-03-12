"""テスト全体で共有されるfixtures。

このモジュールでは、複数のテストファイルで使用される共通のfixtureを定義します。
"""

from dotenv import load_dotenv

load_dotenv()

import sys
import tempfile
from pathlib import Path

from PIL import Image
from pytest_bdd import given

from scorer_wrapper_lib.core.utils import ConsoleLogCapture

# ルートパスの設定
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

# resourcesディレクトリのパス
test_dir = Path(__file__).parent
resources_dir = test_dir / "resources"


# Given ステップ
@given("有効な画像が用意されている", target_fixture="single_image")
def given_single_image() -> list[Image.Image]:
    # 絶対パスで画像ファイルを指定
    image_path = resources_dir / "img" / "1_img" / "file01.webp"
    return [Image.open(image_path)]


@given("複数の有効な画像が用意されている", target_fixture="images")
def given_image_list() -> list[Image.Image]:
    # 絶対パスで画像ファイルを指定
    image_path1 = resources_dir / "img" / "1_img" / "file01.webp"
    image_path2 = resources_dir / "img" / "1_img" / "file02.webp"
    return [
        Image.open(image_path1),
        Image.open(image_path2),
    ]


@given("コンソールログキャプチャツールが初期化される")
def init_console_log_capture() -> ConsoleLogCapture:
    """コンソールログキャプチャツールを初期化するステップ。"""
    return ConsoleLogCapture()


@given("ログファイルパスが指定される")
def specify_log_file_path() -> Path:
    """ログファイルパスを指定するステップ。"""
    temp_dir = tempfile.gettempdir()
    log_file = Path(temp_dir) / "console_log_test.log"
    # テスト前にファイルが存在していたら削除
    if log_file.exists():
        log_file.unlink(missing_ok=True)
    return log_file


@given("標準出力のみキャプチャする設定がされる")
def init_stdout_only_capture() -> ConsoleLogCapture:
    """標準出力のみキャプチャする設定を行うステップ。"""
    return ConsoleLogCapture(capture_stderr=False)
