import hashlib
import logging
import time
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, TypedDict
from urllib.parse import urlparse

import numpy as np
import psutil
import requests
import tensorflow as tf  # type: ignore
import torch
from PIL import Image
from tqdm import tqdm  # type: ignore

# ログ設定
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# 設定
config = {
    "model_path": "https://github.com/KichangKim/DeepDanbooru/releases/download/v1-20191108-sgd-e30/deepdanbooru-v1-20191108-sgd-e30.zip",
    "device": "cuda",
}

# 定数
DEFAULT_CACHE_DIR = Path("models")
DEFAULT_TIMEOUT = 30


# 型定義
class PreprocessedImage(TypedDict):
    pixel_values: Optional[torch.Tensor]  # Transformer用
    input_data: Optional[np.ndarray]  # ONNX用


# model_factory によって生成されるモデルのコンポーネントを表す型定義
class ModelComponents(TypedDict, total=False):
    model_path: Optional[str]  # モデルのパス


class PredictionResult(TypedDict):
    caption: Optional[list[str]]
    output_data: Optional[dict[str, dict[str, float]]]


# ダウンロード関連の関数


def perform_download(url: str, target_path: Path, expected_hash: Optional[str] = None) -> None:
    """
    指定されたURLからファイルをダウンロードし、進捗バーを表示します。

    Args:
        url (str): ダウンロード元のURL。
        target_path (Path): ダウンロードしたファイルを保存するパス。
        expected_hash (Optional[str], optional): ファイルのハッシュ確認用。既定値はNone。
    """
    logger.info(f"{url} から {target_path} へダウンロード中")
    response = requests.get(url, stream=True, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    with (
        open(target_path, "wb") as f,
        tqdm(total=total_size, unit="B", unit_scale=True, desc=target_path.name) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def download_file(
    url: str, cache_dir: Path = DEFAULT_CACHE_DIR, expected_hash: Optional[str] = None
) -> Path:
    """
    キャッシュディレクトリを使用して、指定されたURLからファイルをダウンロードします。

    Args:
        url (str): ダウンロード元のURL。
        cache_dir (Path, optional): キャッシュ用ディレクトリ。既定値は DEFAULT_CACHE_DIR。
        expected_hash (Optional[str], optional): ハッシュ確認用。既定値は None。

    Returns:
        Path: ダウンロードまたはキャッシュされたファイルの絶対パス。
    """
    cache_dir.mkdir(exist_ok=True, parents=True)
    filename = Path(urlparse(url).path).name
    if not filename or len(filename) < 5:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        extension = Path(urlparse(url).path).suffix
        filename = f"{url_hash}{extension if extension else '.bin'}"
    file_path = cache_dir / filename
    if not file_path.is_file():
        perform_download(url, file_path, expected_hash)
    return file_path.resolve()


@lru_cache(maxsize=128)
def get_file_path(path_or_url: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """
    指定された文字列がローカルファイルパスかURLかを判別し、対応するファイルパスを返します。

    Args:
        path_or_url (str): ローカルパスまたはURLの文字列。
        cache_dir (Path, optional): ダウンロードファイルのキャッシュディレクトリ。既定値は DEFAULT_CACHE_DIR。

    Returns:
        Path: 解決済みのファイルパス。

    Raises:
        FileNotFoundError: ローカルファイルが存在しない場合に発生。
    """
    parsed = urlparse(path_or_url)
    if parsed.scheme in ("http", "https"):
        return download_file(path_or_url, cache_dir)
    else:
        local_path = Path(path_or_url)
        if local_path.exists():
            return local_path.resolve()
        raise FileNotFoundError(f"ローカルファイル '{path_or_url}' が見つかりません")


# ヘルパー関数: ZIP解凍処理


def _extract_zip(file_path: Path) -> Path:
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


# ヘルパー関数: タグ情報の一括読み込み
def load_file(path_or_url: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """
    ローカルパスまたはURLからファイルを取得し、ZIPアーカイブの場合は解凍してディレクトリパスを返します。

    Args:
        path_or_url (str): ローカルファイルパスまたはURL。
        cache_dir (Path, optional): キャッシュディレクトリ。既定値は DEFAULT_CACHE_DIR。

    Returns:
        Path: ファイルのパス、またはZIPの場合は解凍先ディレクトリのパス。

    Raises:
        RuntimeError: ファイルの取得に失敗した場合に発生。
    """
    try:
        file_path = get_file_path(path_or_url, cache_dir)
    except Exception as e:
        raise RuntimeError(f"'{path_or_url}' からファイルを取得できません: {e}") from e
    if file_path.suffix.lower() == ".zip":
        return _extract_zip(file_path)
    return file_path


# モデルロード関連のクラス
class ModelLoad:
    _MODEL_STATES: dict[str, str] = {}
    _MEMORY_USAGE: dict[str, float] = {}
    _MODEL_LAST_USED: dict[str, float] = {}  # タイムスタンプを記録
    _CACHE_RATIO = 0.5  # システム全体のメモリの何割までキャッシュに使用するか
    _MODEL_SIZES: dict[str, float] = {}  # モデルサイズをキャッシュするためのクラス変数
    logger = logging.getLogger(__name__)

    @staticmethod
    def get_model_size(model_name: str) -> float:
        """モデルの推定メモリ使用量を取得（MB単位）"""
        # クラス変数にキャッシュがあるか確認
        if hasattr(ModelLoad, "_MODEL_SIZES") and model_name in ModelLoad._MODEL_SIZES:
            return ModelLoad._MODEL_SIZES[model_name]

        # 設定ファイルから読み込む
        model_configs = config
        if model_name in model_configs and "estimated_size_gb" in model_configs[model_name]:
            # GBからMBに変換して内部で扱う
            size_mb = float(model_configs[model_name]["estimated_size_gb"]) * 1024

            # クラス変数にもキャッシュ
            if not hasattr(ModelLoad, "_MODEL_SIZES"):
                ModelLoad._MODEL_SIZES = {}
            ModelLoad._MODEL_SIZES[model_name] = size_mb

            logger.debug(
                f"モデル '{model_name}' のサイズをキャッシュから読み込みました: {size_mb / 1024:.3f}GB"
            )
            return size_mb

        return 0.0

    @staticmethod
    def get_max_cache_size(verbose: bool = False) -> float:
        """システムの最大メモリに基づいてキャッシュサイズを計算"""
        total_memory = psutil.virtual_memory().total / (1024 * 1024)  # MB単位
        cache_size = total_memory * ModelLoad._CACHE_RATIO

        # ログ出力はverboseフラグが有効な場合のみ行う
        if verbose:
            available_memory = psutil.virtual_memory().available / (1024 * 1024)
            ModelLoad.logger.info(
                f"システム全体のメモリ: {total_memory:.1f}MB, "
                f"現在の空きメモリ: {available_memory:.1f}MB, "
                f"設定キャッシュ容量: {cache_size:.1f}MB"
            )

        return float(cache_size)

    @staticmethod
    def _clear_cache_if_needed(model_name: str, model_size: float) -> None:
        """必要に応じて古いモデルをキャッシュから削除します"""
        max_cache = ModelLoad.get_max_cache_size(verbose=False)  # ログ出力しない
        current_cache_size = sum(ModelLoad._MEMORY_USAGE.values())

        if current_cache_size + model_size <= max_cache:
            return

        # GBに変換して表示
        max_cache_gb = max_cache / 1024
        current_cache_gb = current_cache_size / 1024
        model_size_gb = model_size / 1024

        ModelLoad.logger.warning(
            f"キャッシュ容量（{max_cache_gb:.3f}GB）を超過します。"
            f"現在の使用量: {current_cache_gb:.3f}GB + 新規: {model_size_gb:.3f}GB"
        )

        # 使用時刻でソートし、古いものから解放
        models_by_age = sorted(ModelLoad._MODEL_LAST_USED.items(), key=lambda x: x[1])

        for old_model_name, last_used in models_by_age:
            if current_cache_size + model_size <= max_cache:
                break

            if old_model_name == model_name:
                continue  # 現在キャッシュしようとしているモデルはスキップ

            freed_memory = ModelLoad._MEMORY_USAGE.get(old_model_name, 0)
            ModelLoad.logger.info(
                f"モデル '{old_model_name}' を解放します"
                f"（最終使用: {time.strftime('%H:%M:%S', time.localtime(last_used))}, "
                f"解放メモリ: {freed_memory:.1f}MB）"
            )

            # ここで静的解析エラーが出ている
            ModelLoad._MODEL_STATES.pop(old_model_name, None)  # release_modelの代わり
            ModelLoad._MEMORY_USAGE.pop(old_model_name, None)
            ModelLoad._MODEL_LAST_USED.pop(old_model_name, None)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            current_cache_size = sum(ModelLoad._MEMORY_USAGE.values())

    @staticmethod
    def cache_to_main_memory(model_name: str, components: dict[str, Any]) -> dict[str, Any]:
        """メモリ管理を行いながらモデルをキャッシュ"""
        if model_name in ModelLoad._MODEL_STATES and ModelLoad._MODEL_STATES[model_name] == "on_cpu":
            ModelLoad.logger.debug(f"モデル '{model_name}' は既にCPUにあります。")
            ModelLoad._MODEL_LAST_USED[model_name] = time.time()
            return components

        # モデルサイズを取得（すでに計算済みの想定）
        model_size = ModelLoad.get_model_size(model_name)
        # GBに変換して表示
        model_size_gb = model_size / 1024
        ModelLoad.logger.debug(f"モデル '{model_name}' のキャッシュサイズ: {model_size_gb:.3f}GB")

        # キャッシュ容量を確認し必要なら古いモデルを解放
        ModelLoad._clear_cache_if_needed(model_name, model_size)

        # モデルをCPUに移動
        try:
            for component_name, component in components.items():
                if component_name == "pipeline":
                    if hasattr(component, "model"):
                        component.model.to("cpu")
                elif hasattr(component, "to"):
                    component.to("cpu")

            ModelLoad._MODEL_STATES[model_name] = "on_cpu"
            ModelLoad._MEMORY_USAGE[model_name] = model_size
            ModelLoad._MODEL_LAST_USED[model_name] = time.time()

            # ここでのみメモリログを出力
            max_cache = ModelLoad.get_max_cache_size(verbose=True)
            ModelLoad.logger.info(
                f"モデル '{model_name}' をキャッシュしました "
                f"（サイズ: {model_size_gb:.3f}GB, "
                f"現在のキャッシュ使用量: {sum(ModelLoad._MEMORY_USAGE.values()):.1f}MB/{max_cache:.1f}MB）"
            )

            return components

        except Exception as e:
            ModelLoad.logger.error(f"モデルのキャッシュに失敗しました: {str(e)}")
            return components

    @staticmethod
    def _calculate_and_save_model_size(model_name: str, model_size: float) -> None:
        """モデルサイズを計算し、キャッシュとTOMLファイルに保存します"""
        # クラス変数にキャッシュ
        if not hasattr(ModelLoad, "_MODEL_SIZES"):
            ModelLoad._MODEL_SIZES = {}
        ModelLoad._MODEL_SIZES[model_name] = model_size

        # GBに変換して表示
        model_size_gb = model_size / 1024
        ModelLoad.logger.info(f"モデル '{model_name}' の推定サイズを計算しました: {model_size_gb:.3f}GB")

    @staticmethod
    def load_tensorflow_components(
        model_name: str,
        model_path: str,
        device: str,
        model_format: str = "h5",
        extra_metadata_loader: Optional[callable] = None,
    ) -> Optional[dict[str, Any]]:
        """汎用的なTensorflowモデルローダー"""
        if model_name in ModelLoad._MODEL_STATES:
            ModelLoad.logger.debug(f"モデル '{model_name}' は既にロード済みです。")
            return None

        try:
            components = {}
            model_dir = load_file(model_path)

            # モデル形式に応じたロード
            if model_format == "h5":
                h5_model = next(model_dir.glob("*.h5"), None)
                if not h5_model:
                    raise FileNotFoundError(f"H5モデルファイルが見つかりません: {model_dir}")
                ModelLoad.logger.info(f"Tensorflowモデルをロード中: {h5_model}")
                components["model"] = tf.keras.models.load_model(h5_model, compile=True)
            elif model_format == "saved_model":
                ModelLoad.logger.info(f"SavedModelをロード中: {model_dir}")
                components["model"] = tf.saved_model.load(str(model_dir))
            elif model_format == "pb":
                pb_model = next(model_dir.glob("*.pb"), None)
                if not pb_model:
                    raise FileNotFoundError(f"PBモデルファイルが見つかりません: {model_dir}")

            # サイズ計算は一度だけ、かつ結果を再利用
            if model_name not in ModelLoad._MODEL_SIZES:
                model_size = ModelLoad._calculate_tensorflow_size(str(model_dir))
                ModelLoad._MODEL_SIZES[model_name] = model_size
                # 計算時のみログ出力
                model_size_gb = model_size / 1024
                ModelLoad.logger.info(
                    f"モデル '{model_name}' の推定サイズを計算しました: {model_size_gb:.3f}GB"
                )

            # モデル状態の更新
            ModelLoad._MODEL_STATES[model_name] = f"on_{device}"

            # モデルディレクトリを保存
            components["model_dir"] = model_dir

            # 追加のメタデータ読み込み処理（呼び出し元で提供）
            if extra_metadata_loader:
                extra_components = extra_metadata_loader(model_dir)
                if extra_components:
                    components.update(extra_components)

            return components
        except Exception as e:
            ModelLoad.logger.error(f"モデル '{model_name}' のロードに失敗しました: {e}")
            return None

    @staticmethod
    def restore_model_to_cuda(model_name: str, device: str, model: dict[str, Any]) -> dict[str, Any]:
        """モデルを指定 CUDA に復元し、メモリ不足をハンドリングします。"""
        current_state = ModelLoad._MODEL_STATES.get(model_name)
        target_state = f"on_{device}"

        if current_state == target_state:
            ModelLoad.logger.debug(f"モデル '{model_name}' は既に {device} にあります。")
            return model

        if current_state == "on_cpu" and "cuda" in device:
            try:
                for _, component in model.items():
                    if hasattr(component, "to") and callable(component.to):
                        component.to(device)

                ModelLoad._MODEL_STATES[model_name] = target_state
                ModelLoad.logger.debug(
                    f"モデル '{model_name}' をメインメモリから {device} へ復元しました。"
                )
                return model
            except torch.OutOfMemoryError as e:
                error_message = f"CUDAメモリ不足: モデル '{model_name}' の {device} への復元中"
                ModelLoad.logger.error(error_message)
                ModelLoad.logger.error(f"元のPyTorchエラー: {e}")  # 元のエラー表示
                try:
                    if device.startswith("cuda") and torch.cuda.is_available():
                        ModelLoad.logger.error(
                            torch.cuda.memory_summary(device=device)
                        )  # メモリサマリー表示
                except Exception as mem_e:
                    ModelLoad.logger.error(f"CUDAメモリサマリーの取得に失敗: {mem_e}")
                # 状態は変更しない
                raise OutOfMemoryError(error_message) from e
        elif current_state is None:
            ModelLoad.logger.warning(f"モデル '{model_name}' の状態が不明なため、復元できません。")
            return model
        else:
            ModelLoad.logger.debug(
                f"モデル '{model_name}' は復元不要または未サポートの状態です (現在: {current_state}, 要求: {device})。"
            )
            return model

    @staticmethod
    def load_tags(tags_path: str) -> list[str]:
        """
        タグファイルから、空行を除いたタグのリストを読み込みます。

        Args:
            tags_path (str): タグファイルのパス。

        Returns:
            list[str]: タグのリスト。
        """
        with open(tags_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def _calculate_tensorflow_size(model_dir: str) -> float:
        """Tensorflowモデル(.h5ファイル)のメモリ使用量を計算（MB単位）"""
        # モデルディレクトリからh5ファイルを探す
        model_path = Path(model_dir)
        h5_files = list(model_path.glob("*.h5"))

        if not h5_files:
            return 0.0

        h5_file = h5_files[0]
        # ファイルサイズをベースにした推定
        file_size = h5_file.stat().st_size / (1024 * 1024)  # MB単位

        # Tensorflowモデルはメモリ上では約1.2倍のサイズになることが多い
        return file_size * 1.2


# タグの信頼度情報
class TagConfidence(TypedDict):
    """タグの信頼度情報を表す型定義"""

    confidence: float  # 信頼度
    source: str  # タグの情報元（例: deepdanbooru）


class BaseTagger(ABC):
    # チャンクサイズのデフォルト値をクラス変数として定義
    # TODO; デフォルトのチャンクサイズをスペックに応じて動的に決める処理は気が向いたら
    # スペックを参照する処理はmodel_factory にあるのでそれらをまとめて別のモジュールに定義するかも
    DEFAULT_CHUNK_SIZE = 8

    def __init__(self, model_name: str):
        """BaseTagger を初期化します。

        Args:
            model_name (str): モデルの名前。
        """
        self.model_name = model_name
        self.config: dict[str, Any] = config

        self.model_path = self.config["model_path"]
        self.device = self.config.get("device", "cuda")
        # 設定ファイルからチャンクサイズを読み込む (なければデフォルト値を使用)
        self.chunk_size = int(self.config.get("chunk_size", self.DEFAULT_CHUNK_SIZE))
        if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
            # logger.warning(f"設定された chunk_size '{self.chunk_size}' は無効です。デフォルト値 {self.DEFAULT_CHUNK_SIZE} を使用します。") # ログは任意
            self.chunk_size = self.DEFAULT_CHUNK_SIZE

        self.components: dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def __enter__(self) -> "BaseTagger":
        pass

    @abstractmethod
    def __exit__(
        self,
        exception_type: Optional[type[BaseException]],
        exception_value: Optional[BaseException],
        traceback: Any,
    ) -> None:
        pass

    @torch.no_grad()
    def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """画像リストからタグをチャンクに分割してバッチ予測します。"""
        if not images:
            return []

        all_results: list[dict[str, Any]] = []
        num_images = len(images)
        chunk_size = self.chunk_size  # インスタンス変数から取得

        self.logger.info(
            f"モデル '{self.model_name}' で {num_images} 枚の画像をチャンクサイズ {chunk_size} で処理します。"
        )

        # 画像リストをチャンクに分割してループ処理
        for i in range(0, num_images, chunk_size):
            chunk_images = images[i : i + chunk_size]
            current_chunk_size = len(chunk_images)  # 現在のチャンクの実際のサイズ
            self.logger.debug(
                f"チャンク {i // chunk_size + 1}/{(num_images + chunk_size - 1) // chunk_size} (サイズ: {current_chunk_size}) を処理中..."
            )

            try:
                # --- チャンク単位でバッチ処理 ---
                # 1. 前処理 (サブクラスのバッチ対応メソッドを呼び出す)
                processed_batch = self._preprocess_image(chunk_images)

                # 2. 推論 (サブクラスのバッチ対応メソッドを呼び出す)
                raw_outputs = self._run_inference(processed_batch)

                # 3. フォーマット (サブクラスのバッチ対応メソッドを呼び出す)
                formatted_outputs = self._format_predictions(raw_outputs)

                # 4. タグ生成 (サブクラスのバッチ対応メソッドを呼び出す)
                annotation_lists = self._generate_tags(formatted_outputs)

                # 5. 結果を組み立てて all_results に追加
                if isinstance(annotation_lists, list) and len(annotation_lists) == current_chunk_size:
                    # formatted_outputs の型と長さをチェック
                    if isinstance(formatted_outputs, list) and len(formatted_outputs) == current_chunk_size:
                        # ONNX (list[dict]) or Transformer (list[str])
                        if all(isinstance(item, dict) for item in formatted_outputs) or all(
                            isinstance(item, str) for item in formatted_outputs
                        ):
                            for j in range(current_chunk_size):
                                # _generate_tagsの結果は2次元配列なので、インデックスjの要素を取得
                                tags = annotation_lists[j]
                                all_results.append(self._generate_result(formatted_outputs[j], tags))
                        else:
                            self.logger.error(
                                f"チャンク {i // chunk_size + 1}: 予期しないフォーマット済み出力タイプ: {type(formatted_outputs[0]) if formatted_outputs else 'N/A'}"
                            )
                    else:
                        self.logger.error(
                            f"チャンク {i // chunk_size + 1}: フォーマット済み出力の長さが不正です。期待: {current_chunk_size}, 実際: {len(formatted_outputs)}"
                        )

                else:
                    self.logger.error(
                        f"チャンク {i // chunk_size + 1}: タグリストの形式または長さが不正です。期待: {current_chunk_size}, 実際: {len(annotation_lists) if isinstance(annotation_lists, list) else 'N/A'}"
                    )

            except OutOfMemoryError as e:
                self.logger.error(
                    f"チャンク {i // chunk_size + 1} (サイズ: {current_chunk_size}) の処理中にメモリ不足エラーが発生: {e}"
                )
                self.logger.error(
                    "メモリが不足しています。設定ファイルで chunk_size を小さくすることを検討してください。"
                )
                raise OutOfMemoryError(
                    f"チャンク処理中にメモリ不足 (チャンクサイズ: {current_chunk_size})"
                ) from e
            except ValueError as e:  # preprocess などで発生する可能性のある他のエラー
                self.logger.error(f"チャンク {i // chunk_size + 1} の処理中にエラーが発生: {e}")
                raise
            except Exception as e:
                self.logger.exception(f"チャンク {i // chunk_size + 1} の処理中に予期せぬエラーが発生: {e}")
                raise

        self.logger.info(
            f"モデル '{self.model_name}' の全チャンク処理が完了しました。合計 {len(all_results)} 件の結果を生成しました。"
        )
        return all_results

    @abstractmethod
    def _preprocess_image(self, images: list[Image.Image]) -> list[Any]:
        """画像バッチを前処理します。

        Args:
            images: 処理する画像のリスト

        Returns:
            list[Any]: 前処理された画像データのリスト
        """
        pass

    @abstractmethod
    def _run_inference(self, processed_batch: Any) -> Any:
        """モデル推論をバッチで実行します。"""
        pass

    @abstractmethod
    def _format_predictions(self, raw_outputs: Any) -> list[dict[str, dict[str, float]]] | list[str]:
        """モデルのバッチ生出力をフォーマットします。"""
        pass

    @abstractmethod
    def _generate_tags(self, formatted_outputs: Any) -> list[list[str]]:
        """フォーマットされたバッチ出力からタグリストの2次元配列を生成します。
        各画像に対するタグのリストを含むリストを返します。

        Returns:
            list[list[str]]: 各画像のタグリストを含む2次元配列
        """
        pass

    def _generate_result(self, model_output: Any, annotation_list: list[str]) -> dict[str, Any]:
        """標準化された結果の辞書を生成します。

        Args:
            model_name (str): モデルの名前。
            model_output: モデルの出力。
            annotation_list (list[str]): 生成されたタグのリスト。

        Returns:
            dict: モデル出力、モデル名、タグリストを含む辞書。
        """
        return {
            "model_name": self.model_name,
            "model_output": model_output,
            "annotation": annotation_list,
        }


# Tensor flowModel 用中間クラス
class TensorflowModel(BaseTagger, ABC):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # Tensorflow共通の設定
        self.tf_config = tf.compat.v1.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True  # メモリ使用量を動的に確保

    def __enter__(self) -> "TensorflowModel":
        """
        共通のTensorflowモデルローディング処理を実装
        サブクラスは_load_model_componentsをオーバーライドして実際のロード処理を実装する
        """
        components = self._load_model_components(self.model_name, self.model_path, self.device)
        if components:
            self.components = ModelLoad.restore_model_to_cuda(self.model_name, self.device, components)
            self.logger.info(f"モデル '{self.model_name}' を正常にロードしました")
        else:
            self.logger.error(f"モデル '{self.model_name}' のロードに失敗しました")
        return self

    def __exit__(
        self,
        exception_type: Optional[type[BaseException]],
        exception_value: Optional[BaseException],
        traceback: Any,
    ) -> None:
        """共通の終了処理"""
        if hasattr(self, "components") and self.components:
            self.components = ModelLoad.cache_to_main_memory(self.model_name, self.components)

    @abstractmethod
    def _load_model_components(
        self, model_name: str, model_path: str, device: str
    ) -> Optional[dict[str, Any]]:
        """
        具体的なモデルのロード処理はサブクラスで実装
        """
        pass

    def _run_inference(self, processed: list[np.ndarray]) -> tf.Tensor:
        """バッチ処理のための共通推論処理"""
        if all(processed_image.ndim == 3 for processed_image in processed):
            processed_batch = np.stack(processed)
        else:
            raise ValueError(f"予期しない入力形式: {[image.shape for image in processed]}")

        # サブクラスでオーバーライド可能
        return self._tf_model_predict(processed_batch)

    def _tf_model_predict(self, batch_input: np.ndarray) -> tf.Tensor:
        """
        Tensorflowモデルの予測処理
        """
        if "model" not in self.components:
            raise ValueError(f"モデル '{self.model_name}' のcomponentsに'model'が見つかりません")
        return self.components["model"](batch_input)


# DeepDanbooruを用いた具体的なタグ予測クラス
class DeepDanbooruTagger(TensorflowModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.threshold = 0.7  # DeepDanbooru独自の閾値

    def _load_model_components(
        self, model_name: str, model_path: str, device: str
    ) -> Optional[dict[str, Any]]:
        """DeepDanbooru固有のモデルロード処理"""

        def load_deepdanbooru_metadata(model_dir: Path) -> dict[str, Any]:
            try:
                return {
                    "tags": ModelLoad.load_tags(str(model_dir / "tags.txt")),
                    "tags_character": ModelLoad.load_tags(str(model_dir / "tags-character.txt")),
                    "tags_general": ModelLoad.load_tags(str(model_dir / "tags-general.txt")),
                }
            except Exception as e:
                self.logger.error(f"タグファイルの読み込みに失敗しました: {e}")
                return {}

        return ModelLoad.load_tensorflow_components(
            model_name=model_name,
            model_path=model_path,
            device=device,
            model_format="h5",
            extra_metadata_loader=load_deepdanbooru_metadata,
        )

    def _preprocess_image(self, images: list[Image.Image]) -> list[np.ndarray]:
        """DeepDanbooru固有の前処理 - モデルバージョンに合わせて入力サイズを調整"""
        results = []
        # モデル名からバージョンを判断
        if "v1-" in self.model_name:
            target_size = (299, 299)  # v1モデル用サイズ
        else:
            target_size = (512, 512)  # v3/v4モデル用サイズ

        for image in images:
            # 変数名を変えて型の混乱を避ける
            img_resized = image.convert("RGB").resize(target_size)
            img_array = np.asarray(img_resized).astype(np.float32) / 255.0
            results.append(img_array)
        return results

    def _format_predictions(self, raw_output: tf.Tensor) -> list[dict[str, dict[str, float]]] | list[str]:
        """DeepDanbooru固有の結果フォーマット処理"""
        # 現在の実装をそのまま使用
        batch_size = raw_output.shape[0]
        results = []

        for i in range(batch_size):
            single_output = raw_output[i : i + 1]
            general = self._format_general_tags(single_output)
            character = self._format_character_tags(single_output)
            other = self._format_all_tags(single_output)

            results.append(
                {
                    "general": dict(
                        sorted(general.items(), key=lambda x: x[1]["confidence"], reverse=True)
                    ),
                    "character": dict(
                        sorted(character.items(), key=lambda x: x[1]["confidence"], reverse=True)
                    ),
                    "other": dict(sorted(other.items(), key=lambda x: x[1]["confidence"], reverse=True)),
                }
            )

        return results

    def _format_general_tags(self, raw_output: tf.Tensor) -> dict[str, TagConfidence]:
        """
        生の出力から一般タグの情報を抽出し、信頼度情報を付与します。

        Args:
            raw_output (tf.Tensor): モデルからの生出力。

        Returns:
            dict[str, TagConfidence]: 一般タグをキーに、信頼度と情報元を値とする辞書。
        """
        all_tags = self.components["tags"]
        tag_probs = {tag: prob for tag, prob in zip(all_tags, raw_output[0])}
        return {
            tag: TagConfidence(confidence=float(tag_probs[tag]), source="deepdanbooru_general")
            for tag in self.components["tags_general"]
            if tag in tag_probs
        }

    def _format_character_tags(self, raw_output: tf.Tensor) -> dict[str, TagConfidence]:
        """
        生の出力からキャラクタータグの情報を抽出し、信頼度情報を付与します。

        Args:
            raw_output (tf.Tensor): モデルの生出力。

        Returns:
            dict[str, TagConfidence]: キャラクタータグをキーに、信頼度と情報元を値とする辞書。
        """
        all_tags = self.components["tags"]
        tag_probs = {tag: prob for tag, prob in zip(all_tags, raw_output[0])}
        return {
            tag: TagConfidence(confidence=float(tag_probs[tag]), source="deepdanbooru_character")
            for tag in self.components["tags_character"]
            if tag in tag_probs
        }

    def _format_all_tags(self, raw_output: tf.Tensor) -> dict[str, TagConfidence]:
        """
        生の出力から、一般タグ・キャラクタータグ以外のタグ情報を抽出します。

        Args:
            raw_output (tf.Tensor): モデルの生出力。

        Returns:
            dict[str, TagConfidence]: 残りのタグをキーに、信頼度と情報元を値とする辞書。
        """
        all_tags = self.components["tags"]
        tag_probs = {tag: prob for tag, prob in zip(all_tags, raw_output[0])}
        classified = set(self.components["tags_general"]) | set(self.components["tags_character"])
        return {
            tag: TagConfidence(confidence=float(prob), source="deepdanbooru")
            for tag, prob in tag_probs.items()
            if tag not in classified
        }

    def _generate_tags(
        self, formatted_outputs: list[dict[str, dict[str, TagConfidence]]]
    ) -> list[list[str]]:
        """DeepDanbooru固有のタグ生成処理"""
        # 現在の実装をそのまま
        results = []

        for formatted_output in formatted_outputs:
            tags = [
                tag
                for cat, tags in formatted_output.items()
                if cat != "other"
                for tag, conf in tags.items()
                if conf["confidence"] >= self.threshold
            ]
            results.append(tags)

        return results


# 新しい例外クラスを追加
class OutOfMemoryError(Exception):
    """モデルのロードまたは推論時にメモリ不足が発生した場合に送出される例外"""

    pass


if __name__ == "__main__":
    from datetime import datetime

    test_image = Path("tests/resources/img/1_img/file05.webp")
    if not test_image.exists():
        print(f"エラー: テスト画像が見つかりません: {test_image}")
        exit(1)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"deepdanbooru_v1_results_{timestamp}.txt"

    try:
        with DeepDanbooruTagger("deepdanbooru-v1-20191108-sgd-e30") as tagger:
            # 同じ画像を3回複製してバッチテスト
            image = Image.open(test_image)
            test_images = [image] * 3

            predictions = tagger.predict(test_images)  # type: ignore

            with open(result_file, "w", encoding="utf-8") as f:
                f.write("DeepDanbooru V1-20191108 バッチ予測結果\n")
                f.write(f"日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("モデル: deepdanbooru-v1-20191108-sgd-e30\n")
                f.write(f"画像: {test_image} (x{len(test_images)}枚)\n")
                f.write("-" * 80 + "\n\n")

                for i, pred in enumerate(predictions):
                    f.write(f"=== 画像 {i + 1}/{len(predictions)} ===\n")
                    for category, tags in pred["model_output"].items():
                        f.write(f"【{category}】\n")
                        for tag, data in tags.items():
                            if data["confidence"] >= 0.5:
                                f.write(f"{tag:<30} : {data['confidence']:.4f}\n")
                        f.write("\n")
                    f.write(f"検出されたタグ数: {len(pred['annotation'])}\n")
                    f.write("-" * 80 + "\n\n")

            print(f"予測結果を {result_file} に保存しました。")
            print(f"総処理画像数: {len(predictions)}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        exit(1)
