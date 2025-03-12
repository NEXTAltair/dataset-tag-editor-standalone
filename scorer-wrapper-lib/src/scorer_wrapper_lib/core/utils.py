import json
import logging
import os
import sys
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any, Optional, TextIO, Type
from urllib.parse import urlparse

import requests
import toml
from huggingface_hub import snapshot_download
from PIL import Image

logger = logging.getLogger(__name__)


def load_file(path_or_url: str) -> str:
    # TODO: やることが多いので分割
    """
    指定されたパス、URL、または HF Hub のモデル ID からファイルを取得し、ローカルパスを返します。
    - 入力に "http://" または "https://" が含まれている場合は、直リンとしてストリーミングダウンロードします。
    - 入力にスキームがない場合、ローカルファイルとして存在すればそのパスを返し、
      存在しなければ HF Hub のモデル ID とみなし、hf_hub_download を利用してダウンロードします。
    """
    parsed = urlparse(path_or_url)
    if parsed.scheme in ("http", "https"):
        # ダウンロード先フォルダを設定・作成
        cache_dir = Path(__file__).parent / "models"
        cache_dir.mkdir(exist_ok=True)
        filename = Path(parsed.path).name
        local_path = cache_dir / filename
        if not local_path.is_file():
            logger.info(f"Downloading model from {path_or_url} to {local_path}")
            response = requests.get(path_or_url, stream=True)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return str(local_path)
    else:
        # ローカルパスか HF Hub のリポジトリと解釈
        local_path = Path(path_or_url)
        if local_path.exists():
            return str(local_path.resolve())
        else:
            try:
                # HF Hub からリポジトリ全体をダウンロードし、キャッシュディレクトリのパスを返す
                cached_repo_path = snapshot_download(repo_id=path_or_url)
                return cached_repo_path  # キャッシュされたリポジトリのパスを返す
            except Exception as e:
                raise RuntimeError(
                    f"Failed to retrieve file for '{path_or_url}'. "
                    "Please provide a valid local path, a direct URL, or a correct HF Hub repo ID."
                ) from e


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    指定された名前でロガーを初期化します。

    Args:
        name: ロガー名。
        level: ログレベル (デフォルトは logging.INFO)。

    Returns:
        logging.Logger: 設定済みのロガーオブジェクト。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 既にハンドラが設定されている場合は、重複して設定しない
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # 標準出力にログを出力するハンドラ
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # ファイルにログを出力するハンドラ # encoding="utf-8"
        log_file = Path("logs/scorer_wrapper_lib.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


@lru_cache(maxsize=None)
def load_model_config() -> dict[str, Any]:
    """セクションはモデル名 モデルごとのパラメーター

    # TODO: 設定なしのデフォルト値の設定は後でやるが､優先度低い

    Returns:
        dict: model_nameをキーとしたモデルごとのパラメーターの辞書
    """
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "models.toml"
    return toml.load(config_path)


class ConsoleLogCapture:
    """
    コンソール出力（stdout、stderr）をキャプチャするためのコンテキストマネージャー。

    このクラスを with ステートメントで使用すると、そのブロック内のすべての標準出力と
    標準エラー出力をキャプチャし、ログファイルに記録できます。

    Attributes:
        log_file (Path): キャプチャしたログを保存するファイルパス。None を指定した場合はデフォルトパスを使用。
        capture_stdout (bool): 標準出力をキャプチャするかどうか。
        capture_stderr (bool): 標準エラー出力をキャプチャするかどうか。
    """

    def __init__(
        self,
        log_file: Path = Path("logs/console_capture.log"),
        capture_stdout: bool = True,
        capture_stderr: bool = True,
    ):
        """
        コンソールログキャプチャの初期化。

        Args:
            log_file: キャプチャしたログを保存するファイルパス。デフォルトは logs/console_capture.log。
            capture_stdout: 標準出力をキャプチャするかどうか。
            capture_stderr: 標準エラー出力をキャプチャするかどうか。
        """
        self.log_file = log_file
        # ログファイルのディレクトリを作成
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.capture_stdout = capture_stdout
        self.capture_stderr = capture_stderr
        self.stdout_buffer = StringIO() if capture_stdout else None
        self.stderr_buffer = StringIO() if capture_stderr else None
        self._original_stdout: Optional[TextIO] = None
        self._original_stderr: Optional[TextIO] = None

    def __enter__(self) -> "ConsoleLogCapture":
        if self.capture_stdout:
            self._original_stdout = sys.stdout
            sys.stdout = self.stdout_buffer
        if self.capture_stderr:
            self._original_stderr = sys.stderr
            sys.stderr = self.stderr_buffer
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        if self.capture_stdout:
            sys.stdout = self._original_stdout
        if self.capture_stderr:
            sys.stderr = self._original_stderr

        if self.stdout_buffer or self.stderr_buffer:
            with open(self.log_file, "a", encoding="utf-8") as f:
                if self.stdout_buffer:
                    f.write(f"--- STDOUT ---\n{self.stdout_buffer.getvalue()}\n")
                if self.stderr_buffer:
                    f.write(f"--- STDERR ---\n{self.stderr_buffer.getvalue()}\n")


def load_image_from_path(image_path: str) -> Image.Image:
    """
    ファイルパスから画像を読み込む関数

    Args:
        image_path: 画像ファイルのパス

    Returns:
        PIL.Image: 読み込まれた画像オブジェクト

    Raises:
        Exception: 画像の読み込みに失敗した場合
    """
    try:
        image = Image.open(image_path)
        return image
    except Exception as e:
        raise Exception(f"画像の読み込みに失敗しました: {e}")


def get_model_config(model_name: str) -> dict:
    """
    指定されたモデル名の設定を取得します

    Args:
        model_name: 設定を取得するモデルの名前

    Returns:
        dict: モデルの設定情報

    Raises:
        Exception: 設定ファイルの読み込みに失敗した場合
    """
    config_path = f"config/models/{model_name}.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        # 既存のload_model_configの結果からモデル名に対応する設定を取得
        all_configs = load_model_config()
        if model_name in all_configs:
            return all_configs[model_name]
        raise Exception(f"モデル '{model_name}' の設定が見つかりませんでした")


def validate_model_config(config: dict, required_fields: list) -> bool:
    """
    モデル設定が必要なフィールドを含んでいるか検証します

    Args:
        config: 検証するモデル設定
        required_fields: 必須フィールドのリスト

    Returns:
        bool: 全ての必須フィールドが含まれている場合はTrue

    Raises:
        Exception: 必須フィールドが不足している場合
    """
    for field in required_fields:
        if field not in config:
            raise Exception(f"モデル設定に必須フィールド '{field}' がありません")
    return True
