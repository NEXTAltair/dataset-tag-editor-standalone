import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
import toml
from tqdm import tqdm

logger = logging.getLogger(__name__)


CONFIG_TOML = Path("config") / "models.toml"
LOG_FILE = Path("logs/scorer_wrapper_lib.log")
DEFAULT_CACHE_DIR = Path("models")
DEFAULT_TIMEOUT = 30


def _get_cache_path(url: str, cache_dir: Path) -> Path:
    """URLからキャッシュファイルパスを生成する"""
    filename = Path(urlparse(url).path).name
    if not filename or len(filename) < 5:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        extension = Path(urlparse(url).path).suffix
        filename = f"{url_hash}{extension}" if extension else f"{url_hash}.bin"
    return cache_dir / filename


def _is_cached(url: str, cache_dir: Path) -> tuple[bool, Path]:
    """URLに対応するファイルがキャッシュに存在するか確認する"""
    local_path = _get_cache_path(url, cache_dir)
    return local_path.is_file(), local_path


def _perform_download(url: str, target_path: Path, expected_hash: Optional[str] = None) -> None:
    """実際のダウンロード処理を行う（進捗表示付き）"""
    logger.info(f"Downloading model from {url} to {target_path}")
    response = requests.get(url, stream=True, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()

    # ファイルサイズを取得
    total_size = int(response.headers.get("content-length", 0))

    with open(target_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=target_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def _download_from_url(
    url: str, cache_dir: Path = DEFAULT_CACHE_DIR, expected_hash: Optional[str] = None
) -> Path:
    """
    URLからファイルをダウンロードし、キャッシュされたローカルパスを返します。

    Args:
        url: ダウンロードするファイルのURL
        cache_dir: キャッシュディレクトリ（デフォルト: 設定ファイルから）
        expected_hash: 期待されるSHA256ハッシュの先頭部分（オプション）

    Returns:
        Path: 絶対パスに変換されたパスオブジェクト
    """
    # ダウンロード先フォルダを作成
    cache_dir.mkdir(exist_ok=True, parents=True)

    # キャッシュチェック
    is_cached, local_path = _is_cached(url, cache_dir)

    # キャッシュされていなければダウンロード
    if not is_cached:
        _perform_download(url, local_path, expected_hash)

    return local_path.resolve()


@lru_cache(maxsize=128)
def get_file_path(path_or_url: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """パスまたはURLからローカルファイルパスを取得（結果をキャッシュ）"""
    parsed = urlparse(path_or_url)

    if parsed.scheme in ("http", "https"):
        return _download_from_url(path_or_url, cache_dir)
    else:
        return _get_local_file_path(path_or_url)


def _get_local_file_path(path: str) -> Path:
    """ローカルファイルパスを検証し、絶対パスを返します。"""
    local_path = Path(path)
    if local_path.exists():
        return local_path.resolve()
    raise FileNotFoundError(f"ローカルファイル '{path}' が見つかりません")


def load_file(path_or_url: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> str:
    """
    指定されたパスまたはURLからファイルを取得し、ローカルパスを返します。

    Args:
        path_or_url: ローカルパスまたはURL
        cache_dir: キャッシュディレクトリ（オプション）

    Returns:
        str: ローカルファイルへのパス

    Raises:
        RuntimeError: ファイルの取得に失敗した場合
    """
    try:
        return str(get_file_path(path_or_url, cache_dir))
    except requests.RequestException as e:
        raise RuntimeError(f"URLからのダウンロードに失敗しました: {e}") from e
    except FileNotFoundError as e:
        raise RuntimeError(f"ローカルファイルが見つかりません: {e}") from e
    except Exception as e:
        raise RuntimeError(
            f"'{path_or_url}' からのファイル取得に失敗しました。"
            "有効なローカルパス、または直接URLを指定してください。"
            f"エラー詳細: {e}"
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
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


@lru_cache(maxsize=None)
def load_model_config() -> dict[str, dict[str, Any]]:
    """セクションはモデル名 モデルごとのパラメーター

    # NOTE: 設定なしのデフォルト値の設定はBaseScorerでやる

    Returns:
        dict[str, dict[str, Any]]: model_nameをキーとしたモデルごとのパラメーターの辞書
    """
    # ファイルの内容を読み込む
    config_data = toml.load(CONFIG_TOML)

    if not isinstance(config_data, dict):
        raise TypeError("構成データは辞書である必要があります")
    return dict(config_data)
