import gc  # release_onnx_components で使うので import をファイルの先頭に移動
import logging
import time
from pathlib import Path
from typing import Any, Optional

import onnxruntime as ort
import psutil
import tensorflow as tf
import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)

from ..exceptions.model_errors import OutOfMemoryError  # インポート
from . import utils

logger = logging.getLogger(__name__)


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
        model_configs = utils.load_model_config()
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
    def get_max_cache_size() -> float:
        """システムの最大メモリに基づいてキャッシュサイズを計算"""
        total_memory = psutil.virtual_memory().total / (1024 * 1024)  # MB単位
        cache_size = total_memory * ModelLoad._CACHE_RATIO

        # 現在のメモリ使用状況もログ出力
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
        max_cache = ModelLoad.get_max_cache_size()
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

            ModelLoad.release_model(old_model_name)
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
        ModelLoad.logger.info(f"モデル '{model_name}' の推定サイズ: {model_size_gb:.3f}GB")

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

            max_cache = ModelLoad.get_max_cache_size()
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

        # TOMLファイルにも保存
        utils.save_model_size(model_name, model_size)

        # GBに変換して表示
        model_size_gb = model_size / 1024
        ModelLoad.logger.info(f"モデル '{model_name}' の推定サイズを計算しました: {model_size_gb:.3f}GB")

    @staticmethod
    def load_transformer_components(
        model_name: str, model_path: str, device: str
    ) -> Optional[dict[str, Any]]:
        """Transformerモデルのコンポーネントをロードし、メモリ不足をハンドリングします。"""
        if model_name in ModelLoad._MODEL_STATES:
            ModelLoad.logger.debug(f"モデル '{model_name}' は既に読み込まれています。")
            return None

        try:
            # 適切なプロセッサとモデルを自動的に選択
            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)

            components = {"model": model, "processor": processor}

            # モデルサイズの計算と保存（ロード時に実行）
            if not hasattr(ModelLoad, "_MODEL_SIZES") or model_name not in ModelLoad._MODEL_SIZES:
                model_size = ModelLoad._calculate_transformer_size(model)
                ModelLoad._calculate_and_save_model_size(model_name, model_size)

            ModelLoad._MODEL_STATES[model_name] = f"on_{device}"
            return components

        except torch.OutOfMemoryError as e:
            # メモリ不足エラーハンドリングを base.py から移動
            error_message = f"CUDAメモリ不足: モデル '{model_name}' のロード中 (デバイス: {device})"
            ModelLoad.logger.error(error_message)
            ModelLoad.logger.error(f"元のPyTorchエラー: {e}")
            try:
                if device.startswith("cuda") and torch.cuda.is_available():
                    ModelLoad.logger.error(torch.cuda.memory_summary(device=device))  # メモリサマリー表示
            except Exception as mem_e:
                ModelLoad.logger.error(f"CUDAメモリサマリーの取得に失敗: {mem_e}")
            if model_name in ModelLoad._MODEL_STATES:
                del ModelLoad._MODEL_STATES[model_name]
            raise OutOfMemoryError(error_message) from e
        # その他のエラーはそのまま送出

    @staticmethod
    def load_onnx_components(model_name: str, model_repo: str, device: str) -> dict[str, Any]:
        """ONNXモデルのコンポーネントをロードし、メモリ不足をハンドリング。"""
        # NOTE: _MODEL_STATES でモデルの状態を管理しない｡ メインメモリーに移動はONNXの仕様上不可能

        session = None
        try:
            # ONNXランタイムセッションの作成
            csv_path, model_path = utils.download_onnx_tagger_model(model_repo)

            # 利用可能なプロバイダーを取得
            available_providers = ort.get_available_providers()
            ModelLoad.logger.debug(f"利用可能なプロバイダー: {available_providers}")

            # デバイスに基づいてプロバイダーを選択
            if device == "cuda" and "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

            ModelLoad.logger.debug(f"ONNXモデル '{model_path}' をロードしています...")
            session = ort.InferenceSession(model_path, providers=providers)

            components = {"session": session, "csv_path": csv_path}

            if model_name not in ModelLoad._MODEL_SIZES:
                # ONNXモデルの倍率は1.5
                model_size = ModelLoad._calculate_model_size(model_path, 1.5)
                # モデルサイズを計算して保存
                ModelLoad._calculate_and_save_model_size(model_name, model_size)
                ModelLoad.logger.info(f"モデル '{model_name}' の推定サイズ: {model_size / 1024:.3f}GB")

            return components

        except ort.capi.onnxruntime_pybind11_state.RuntimeException as e:
            if "Failed to allocate memory" in str(e) or "CUDA error" in str(e):
                error_message = f"ONNX Runtime メモリ/CUDAエラー: モデル '{model_name}' のロード中"
                ModelLoad.logger.error(error_message)
                ModelLoad.logger.error(f"元のONNX Runtimeエラー: {e}")
                raise OutOfMemoryError(error_message) from e
            else:
                raise
        except Exception as e:
            ModelLoad.logger.exception(f"ONNXモデル '{model_name}' のロード中に予期せぬエラーが発生: {e}")
            raise

    @staticmethod
    def load_tensorflow_components(
        model_name: str,
        model_path: str,
        device: str,
        model_format: str = "h5",
    ) -> dict[str, Any]:
        """汎用的なTensorflowモデルローダー"""
        try:
            components = {}
            model_dir = utils.load_file(model_path)
            model_file_path = None

            if model_format == "h5":
                model_file_path = next(model_dir.glob("*.h5"), None)
                if not model_file_path:
                    raise FileNotFoundError(f"H5モデルファイルが見つかりません: {model_dir}")
                multiplier = 1.2
                ModelLoad.logger.info(f"Tensorflowモデルをロード中: {model_file_path}")
                components["model"] = tf.keras.models.load_model(model_file_path, compile=True)

            elif model_format == "saved_model":
                model_file_path = model_dir  # ディレクトリ全体
                multiplier = 1.3
                ModelLoad.logger.info(f"SavedModelをロード中: {model_dir}")
                components["model"] = tf.saved_model.load(model_dir)

            elif model_format == "pb":
                pb_model = next(model_dir.glob("*.pb"), None)
                if not pb_model:
                    raise FileNotFoundError(f"PBモデルファイルが見つかりません: {model_dir}")
                model_file_path = pb_model
                multiplier = 1.3
                ModelLoad.logger.info(f"PBモデルをロード中: {pb_model}")
                components["model"] = tf.saved_model.load(model_dir)

            # サイズ計算（テレメトリのためだけに使用）
            if model_name not in ModelLoad._MODEL_SIZES and model_file_path:
                model_size = ModelLoad._calculate_model_size(model_file_path, multiplier)
                ModelLoad._MODEL_SIZES[model_name] = model_size
                ModelLoad.logger.info(f"モデル '{model_name}' の推定サイズ: {model_size / 1024:.3f}GB")

            # モデルディレクトリを保存
            components["model_dir"] = model_dir

            return components
        except Exception as e:
            ModelLoad.logger.error(f"モデル '{model_name}' のロードに失敗しました: {e}")
            raise

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
    def release_model(model_name: str) -> None:
        """
        モデルを _LOADED_SCORERS から削除 メモリから解放し、GPU キャッシュをクリアします。
        """
        if model_name in ModelLoad._MODEL_STATES:
            del ModelLoad._MODEL_STATES[model_name]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ModelLoad.logger.debug(f"モデル '{model_name}' を解放しました。")

    @staticmethod
    def release_model_components(model_name: str, components: dict[str, Any]) -> dict[str, Any]:
        """
        ONNXのsessionまたはTensorflowのmodelを解放
        モデルコンポーネントのリソースを解放
        """
        try:
            for key in ["session", "model"]:
                if key in components and components[key] is not None:
                    # 参照を保持してから削除
                    component = components[key]
                    components[key] = None

                    # クローズメソッドがあれば呼び出す
                    if hasattr(component, "close") and callable(component.close):
                        component.close()

                    # 明示的に参照を削除
                    del component

            # GCを実行
            gc.collect()

            return components
        except Exception as e:
            ModelLoad.logger.error(f"モデル '{model_name}' のリソース解放中にエラー発生: {e}")
            return components

    @staticmethod
    def _calculate_transformer_size(model: torch.nn.Module) -> float:
        """Transformerモデルのメモリ使用量を計算（MB単位）"""
        # パラメータサイズを計算
        total_params = sum(p.numel() for p in model.parameters())
        # 4バイト（float32）× パラメータ数 → MB単位に変換
        param_size = total_params * 4 / (1024 * 1024)

        # パラメータ以外のオーバーヘッドを考慮して20%上乗せ
        return param_size * 1.2

    @staticmethod
    def _calculate_model_size(model_file_path: Path, multiplier: float) -> float:
        """モデルのメモリ使用量を計算（MB単位）

        Args:
            model_file_path: モデル本体のファイルパス
            multiplier: メモリーに展開したときにどれくらいの倍率になるか

        Returns:
            float: 推定メモリ使用量（MB単位）
        """
        # ファイルかディレクトリかで異なる計算
        if model_file_path.is_file():
            file_size = model_file_path.stat().st_size / (1024 * 1024)  # MB単位
        elif model_file_path.is_dir():  # saved_modelの場合はディレクトリ
            file_size = sum(f.stat().st_size for f in model_file_path.glob("**/*") if f.is_file()) / (
                1024 * 1024
            )
        else:
            return 0.0

        return file_size * multiplier
