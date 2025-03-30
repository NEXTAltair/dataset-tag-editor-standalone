import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import huggingface_hub
import requests
import toml
from tqdm import tqdm

logger = logging.getLogger(__name__)


CONFIG_TOML = Path("config") / "taggers.toml"
LOG_FILE = Path("logs/tagger_wrapper_lib.log")
DEFAULT_CACHE_DIR = Path("models")
DEFAULT_TIMEOUT = 30
WD_MODEL_FILENAME = "model.onnx"
WD_LABEL_FILENAME = "selected_tags.csv"


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


def extract_zip(file_path: Path) -> Path:
    """
    ZIPアーカイブを解凍して、その解凍先ディレクトリのパスを返します。

    Args:
        file_path (Path): ZIPファイルのパス。

    Returns:
        Path: 解凍先ディレクトリのパス。
    """
    import zipfile

    extract_dir = file_path.parent / file_path.stem
    if not extract_dir.exists():
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
    return extract_dir


def load_file(path_or_url: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """
    指定されたパスまたはURLからファイルを取得し、ローカルパスを返します。
    ZIPファイルの場合は解凍して、そのディレクトリのパスを返します。

    Args:
        path_or_url: ローカルパスまたはURL
        cache_dir: キャッシュディレクトリ（オプション）

    Returns:
        Path: ローカルファイルへのパス、またはZIPの場合は解凍先ディレクトリのパス

    Raises:
        RuntimeError: ファイルの取得に失敗した場合
    """
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

    # CSVファイルを検索（最初に見つかったものを使用）
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

    # NOTE: 設定なしのデフォルト値の設定はBaseTaggerでやる

    Returns:
        dict[str, dict[str, Any]]: model_nameをキーとしたモデルごとのパラメーターの辞書
    """
    # ファイルの内容を読み込む
    config_data = toml.load(CONFIG_TOML)

    if not isinstance(config_data, dict):
        raise TypeError("構成データは辞書である必要があります")
    return dict(config_data)


def save_model_size(model_name: str, size_mb: float) -> None:
    """モデルのサイズ推定値をtaggers.tomlに保存する (GB単位)"""
    try:
        # MBからGBに変換
        size_gb = size_mb / 1024

        # 既存のTOMLファイルを読み込む
        if CONFIG_TOML.exists():
            config_data = toml.load(CONFIG_TOML)
        else:
            logger.error(f"設定ファイル {CONFIG_TOML} が見つかりません")
            return

        # モデル設定が存在するか確認
        if model_name not in config_data:
            logger.warning(f"モデル '{model_name}' の設定が見つかりません")
            return

        # サイズ情報を追加/更新 (GB単位)
        config_data[model_name]["estimated_size_gb"] = round(size_gb, 3)  # 小数点3桁まで丸める

        # 変更をファイルに書き込む
        with open(CONFIG_TOML, "w") as f:
            toml.dump(config_data, f)

        logger.debug(f"モデル '{model_name}' の推定サイズ ({size_gb:.3f}GB) を保存しました")
    except Exception as e:
        logger.error(f"モデルサイズの保存に失敗しました: {e}")
