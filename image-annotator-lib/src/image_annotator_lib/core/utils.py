import hashlib
import logging
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import huggingface_hub
import imagehash
import requests
import toml
from PIL import Image
from tqdm import tqdm

DEFAULT_PATHS = {
    "config_toml": Path("config") / "annotator_config.toml",
    "log_file": Path("logs") / "image-annotator-lib.log",
    "cache_dir": Path("models"),
}

DEFAULT_TIMEOUT = 30
WD_MODEL_FILENAME = "model.onnx"
WD_LABEL_FILENAME = "selected_tags.csv"


def setup_logger(
    name: str, level: int = logging.INFO, log_file: Path = DEFAULT_PATHS["log_file"]
) -> logging.Logger:
    """指定された名前でロガーを初期化します。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ロガーの初期化
logger = setup_logger(__name__)


def calculate_phash(image: Image.Image) -> str:
    """画像から pHash を計算します。

    Args:
        image: pHash を計算する PIL Image オブジェクト。

    Returns:
        str: pHash 文字列
    """
    rgb_image = image.convert("RGB")
    hash_val = imagehash.phash(rgb_image)
    return str(hash_val)


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


def _perform_download(url: str, target_path: Path) -> None:
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


def _download_from_url(url: str, cache_dir: Path = DEFAULT_PATHS["cache_dir"]) -> Path:
    """URLからファイルをダウンロードします。"""
    cache_dir.mkdir(exist_ok=True, parents=True)
    is_cached, local_path = _is_cached(url, cache_dir)
    if not is_cached:
        _perform_download(url, local_path)
    return local_path.resolve()


@lru_cache(maxsize=128)
def get_file_path(path_or_url: str, cache_dir: Path = DEFAULT_PATHS["cache_dir"]) -> Path:
    """パスまたはURLからローカルファイルパスを取得します。"""
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


def extract_zip(zip_path: Path) -> Path:
    """
    ZIPファイルを解凍し、解凍先のディレクトリパスを返します。

    Args:
        zip_path: ZIPファイルのパス

    Returns:
        Path: 解凍先ディレクトリのパス

    Raises:
        RuntimeError: 解凍に失敗した場合
    """
    extract_dir = zip_path.parent / zip_path.stem
    try:
        logger.info(f"Extracting ZIP file: {zip_path} to {extract_dir}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        return extract_dir
    except Exception as e:
        raise RuntimeError(f"ZIPファイルの解凍に失敗しました: {e}") from e


def load_file(path_or_url: str, cache_dir: Path = DEFAULT_PATHS["cache_dir"]) -> Path:
    """ファイルを取得し、ローカルパスを返します。"""
    try:
        file_path = get_file_path(path_or_url, cache_dir)
        if file_path.suffix.lower() == ".zip":
            return extract_zip(file_path)
        return file_path
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


def download_onnx_tagger_model(model_repo: str) -> tuple[Path, Path]:
    """WD-Taggerのモデルをダウンロードする"""
    # リポジトリ内のファイル一覧を取得
    repo_files = huggingface_hub.list_repo_files(model_repo)

    # CSVファイルを検索(最初に見つかったものを使用)
    csv_filename = next((f for f in repo_files if f.endswith(".csv")), WD_LABEL_FILENAME)

    csv_path = huggingface_hub.hf_hub_download(
        model_repo,
        csv_filename,
    )

    model_path = huggingface_hub.hf_hub_download(
        model_repo,
        WD_MODEL_FILENAME,
    )

    return Path(csv_path), Path(model_path)


@lru_cache
def load_model_config(config_path: Path = DEFAULT_PATHS["config_toml"]) -> dict[str, dict[str, Any]]:
    """モデル設定を読み込みます。"""
    config_data = toml.load(config_path)
    if not isinstance(config_data, dict):
        raise TypeError("構成データは辞書である必要があります")
    return dict(config_data)


def save_model_size(
    model_name: str, size_mb: float, config_path: Path = DEFAULT_PATHS["config_toml"]
) -> None:
    """モデルのサイズ推定値を保存します。"""
    try:
        size_gb = size_mb / 1024

        if config_path.exists():
            config_data = toml.load(config_path)
        else:
            logger.error(f"設定ファイル {config_path} が見つかりません")
            return

        if model_name not in config_data:
            logger.warning(f"モデル '{model_name}' の設定が見つかりません")
            return

        config_data[model_name]["estimated_size_gb"] = round(size_gb, 3)

        with open(config_path, "w") as f:
            toml.dump(config_data, f)

        logger.debug(f"モデル '{model_name}' の推定サイズ ({size_gb:.3f}GB) を保存しました")
    except Exception as e:
        logger.error(f"モデルサイズの保存に失敗しました: {e}")
