import logging
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import toml

logger = logging.getLogger(__name__)


def _is_cached(url: str, cache_dir: Path) -> tuple[bool, Path]:
    """URLに対応するファイルがキャッシュに存在するか確認する"""
    filename = Path(urlparse(url).path).name
    local_path = cache_dir / filename
    return local_path.is_file(), local_path


def _perform_download(url: str, target_path: Path) -> None:
    """実際のダウンロード処理を行う"""
    logger.info(f"Downloading model from {url} to {target_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def _download_from_url(url: str) -> Path:
    """
    URLからファイルをダウンロードし、キャッシュされたローカルパスを返します。

    Args:
        url: ダウンロードするファイルのURL

    Returns:
        Path: ダウンロードされたファイルのローカルパス
    """
    # ダウンロード先フォルダを設定・作成
    cache_dir = Path(__file__).parent / "models"
    cache_dir.mkdir(exist_ok=True)

    # キャッシュチェック
    is_cached, local_path = _is_cached(url, cache_dir)

    # キャッシュされていなければダウンロード
    if not is_cached:
        _perform_download(url, local_path)

    return local_path


def _get_local_file_path(path: str) -> Path:
    """
    ローカルファイルパスを検証し、絶対パスを返します。

    Args:
        path: 検証するローカルファイルパス

    Returns:
        Path: 絶対パスに変換されたパスオブジェクト

    Raises:
        FileNotFoundError: 指定されたパスにファイルが存在しない場合
    """
    local_path = Path(path)
    if local_path.exists():
        return local_path.resolve()
    raise FileNotFoundError(f"ローカルファイル '{path}' が見つかりません")


def load_file(path_or_url: str) -> str:
    """
    指定されたパスまたはURLからファイルを取得し、ローカルパスを返します。

    - 入力に "http://" または "https://" が含まれている場合は、直リンとしてストリーミングダウンロードします。
    - 入力にスキームがない場合、ローカルファイルとして存在すればそのパスを返します。
    - それ以外の場合はエラーを発生させます。

    Args:
        path_or_url: ローカルパスまたはURL

    Returns:
        str: ローカルファイルへのパス

    Raises:
        RuntimeError: ファイルの取得に失敗した場合
    """
    parsed = urlparse(path_or_url)

    try:
        if parsed.scheme in ("http", "https"):
            # URLからダウンロード
            return str(_download_from_url(path_or_url))
        else:
            # ローカルパスを試す
            try:
                return str(_get_local_file_path(path_or_url))
            except FileNotFoundError:
                # ローカルファイルもURLでもない場合はエラー
                raise RuntimeError(f"'{path_or_url}' は有効なローカルパスでもURLでもありません") from None
    except Exception as e:
        # 全ての例外を捕捉し、わかりやすいエラーメッセージを提供
        raise RuntimeError(
            f"'{path_or_url}' からのファイル取得に失敗しました。"
            "有効なローカルパス、または直接URLを指定してください。"
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
def load_model_config() -> dict[str, dict[str, Any]]:
    """セクションはモデル名 モデルごとのパラメーター

    # TODO: 設定なしのデフォルト値の設定は後でやるが､優先度低い

    Returns:
        dict[str, dict[str, Any]]: model_nameをキーとしたモデルごとのパラメーターの辞書
    """
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "models.toml"
    config_data = toml.load(config_path)
    if not isinstance(config_data, dict):
        raise TypeError("Config data must be a dictionary")
    return dict(config_data)
