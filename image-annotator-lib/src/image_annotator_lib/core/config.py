import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import toml

# --- 定数定義 ---
DEFAULT_PATHS = {
    "config_toml": Path("config") / "annotator_config.toml",
    "log_file": Path("logs") / "image-annotator-lib.log",
    "cache_dir": Path("models"),
}
DEFAULT_TIMEOUT = 30
WD_MODEL_FILENAME = "model.onnx"
WD_LABEL_FILENAME = "selected_tags.csv"

# ロガーの初期化
logger = logging.getLogger(__name__)


@lru_cache
def _load_config_from_file(config_path: Path) -> dict[str, dict[str, Any]]:
    """設定ファイルをTOML形式で読み込む内部ヘルパー関数。"""
    try:
        logger.debug(f"構成ファイルを読み込みます: {config_path}")
        config_data = toml.load(config_path)
        if not isinstance(config_data, dict):
            logger.error(f"設定ファイル {config_path} の形式が不正です。辞書形式である必要があります。")
            raise TypeError("構成データは辞書である必要があります")
        logger.debug(f"構成ファイルを正常に読み込みました: {config_path}")
        return dict(config_data)
    except FileNotFoundError:
        logger.error(f"設定ファイル {config_path} が見つかりません。デフォルト設定で続行します。")
        return {}  # ファイルが見つからない場合は空の辞書を返す
    except Exception as e:
        logger.exception(f"設定ファイル {config_path} の読み込み中に予期せぬエラーが発生しました: {e}")
        raise  # 予期せぬエラーは再送出


def save_model_size(model_name: str, size_mb: float, config_path: Path | None = None) -> None:
    """モデルのサイズ推定値を保存します。"""
    if config_path is None:
        config_path = DEFAULT_PATHS["config_toml"]
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


class ModelConfigRegistry:
    """設定ファイル全体を管理し、設定値へのアクセスを提供します。"""

    def __init__(self) -> None:
        """初期化時に設定データを空の辞書で初期化します。"""
        self._config_data: dict[str, dict[str, Any]] = {}
        self._is_loaded = False

    def load(self, config_path: Path | None = None) -> None:
        """設定ファイルを読み込み、内部データ (_config_data) を更新します。"""
        if config_path is None:
            config_path = DEFAULT_PATHS["config_toml"]

        self._config_data = _load_config_from_file(config_path)
        logger.info(f"設定マネージャーが {config_path} から設定をロードしました。")

    def get(self, model_name: str, key: str, default: Any = None) -> Any | None:
        """指定されたモデルとキーに対応する設定値を取得します。

        前提: `model_name` は設定データ内に必ず存在します。

        - `key` がモデルの設定内に存在する場合: その値を返します。
        - `key` がモデルの設定内に存在しない場合: `default` 引数で指定された値を返します。

        Args:
            model_name: 設定値を取得したいモデルの名前 (設定内に存在することが前提)。
            key: 取得したい設定のキー。
            default: `key` が見つからなかった場合に返すデフォルト値 (デフォルト: None)。

        Returns:
            取得した設定値。`key` が見つからなければ `default` 引数の値。
        """
        model_config = self._config_data[model_name]
        return model_config.get(key, default)

    def get_all_config(self) -> dict[str, Any]:
        """ロード済みの設定データ全体を辞書として返します。"""
        return self._config_data.copy()  # 内部データのコピーを返す


# --- 共有インスタンスの作成と初期ロード --- #
config_registry = ModelConfigRegistry()
try:
    config_registry.load()
except Exception:
    logger.exception("共有設定レジストリの初期ロード中にエラーが発生しました。")
