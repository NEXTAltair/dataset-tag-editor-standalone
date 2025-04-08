import gc
import logging
import time
from pathlib import Path
from typing import Any, ClassVar

import onnxruntime as ort
import psutil
import tensorflow as tf
import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor, CLIPModel, CLIPProcessor, pipeline

from ..exceptions.errors import OutOfMemoryError
from . import utils
from .config import config_registry

logger = logging.getLogger(__name__)


class BaseModelLoader:
    """モデルローダーの基底クラス"""

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self._MODEL_STATES: dict[str, str] = {}
        self._MEMORY_USAGE: dict[str, float] = {}
        self._MODEL_LAST_USED: dict[str, float] = {}
        self._CACHE_RATIO = 0.5
        self._MODEL_SIZES: dict[str, float] = {}
        self.logger = logging.getLogger(__name__)

    def get_model_size(self) -> float:
        """モデルの推定メモリ使用量を取得(MB単位)"""
        if self.model_name in self._MODEL_SIZES:
            return self._MODEL_SIZES[self.model_name]

        # config_registry から estimated_size_gb を取得
        try:
            estimated_size_gb = config_registry.get(self.model_name, "estimated_size_gb")

            size_mb = float(estimated_size_gb) * 1024
            self._MODEL_SIZES[self.model_name] = size_mb
            self.logger.debug(
                f"モデル '{self.model_name}' のサイズをキャッシュから読み込みました: {size_mb / 1024:.3f}GB"
            )
            return size_mb
        except Exception:
            return 0.0

    def get_max_cache_size(self) -> float:
        """システムの最大メモリに基づいてキャッシュサイズを計算"""
        total_memory = psutil.virtual_memory().total / (1024 * 1024)
        cache_size = total_memory * self._CACHE_RATIO
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        self.logger.info(
            f"システム全体のメモリ: {total_memory:.1f}MB, "
            f"現在の空きメモリ: {available_memory:.1f}MB, "
            f"設定キャッシュ容量: {cache_size:.1f}MB"
        )
        return float(cache_size)

    def _clear_cache_if_needed(self, model_size: float) -> None:
        """必要に応じて古いモデルをキャッシュから削除"""
        max_cache = self.get_max_cache_size()
        # current_cache_size はループ内で更新するので、初期値だけ計算
        initial_cache_size = sum(self._MEMORY_USAGE.values())

        if initial_cache_size + model_size <= max_cache:
            return

        max_cache_gb = max_cache / 1024
        current_cache_gb = initial_cache_size / 1024  # 初期値でログ
        model_size_gb = model_size / 1024
        self.logger.warning(
            f"キャッシュ容量({max_cache_gb:.3f}GB)を超過します。"
            f"現在の使用量: {current_cache_gb:.3f}GB + 新規: {model_size_gb:.3f}GB"
        )

        models_by_age = sorted(self._MODEL_LAST_USED.items(), key=lambda x: x[1])

        for old_model_name, last_used in models_by_age:
            # --- 修正点:ループ条件チェックで最新のメモリ使用量を見る ---
            current_cache_size = sum(self._MEMORY_USAGE.values())  # ★毎回ここで最新の値を取得
            if current_cache_size + model_size <= max_cache:
                self.logger.info("必要なキャッシュ容量が確保されたため、解放処理を停止します。")
                break
            # --- 修正点ここまで ---

            if old_model_name == self.model_name:
                continue

            # ... (モデル解放処理) ...
            freed_memory = self._MEMORY_USAGE.get(old_model_name, 0)
            self.logger.info(
                f"モデル '{old_model_name}' を解放します"
                f"(最終使用: {time.strftime('%H:%M:%S', time.localtime(last_used))}, "
                f"解放メモリ: {freed_memory:.1f}MB)"
            )
            self.release_model(old_model_name)
            # current_cache_size の更新は不要(次のループ冒頭で再計算するため)

        # ループ終了後のチェックで最新の値を使うように修正
        final_cache_size = sum(self._MEMORY_USAGE.values())
        if final_cache_size + model_size > max_cache:
            final_cache_gb = final_cache_size / 1024  # finalの値でログ
            self.logger.error(
                f"古いモデルを解放しても、モデル '{self.model_name}' ({model_size_gb:.3f}GB) "
                f"のための十分なキャッシュ容量 ({max_cache_gb:.3f}GB) を確保できませんでした。"
                f"現在の使用量: {final_cache_gb:.3f}GB"
            )

    def release_model(self, model_name: str) -> None:
        """モデルの状態とメモリ使用量の記録を削除"""
        if model_name in self._MODEL_STATES:
            del self._MODEL_STATES[model_name]
        if model_name in self._MEMORY_USAGE:
            del self._MEMORY_USAGE[model_name]
        if model_name in self._MODEL_LAST_USED:
            del self._MODEL_LAST_USED[model_name]

    def _calculate_model_size(self, model_file_path: Path, multiplier: float) -> float:
        """モデルファイルサイズからメモリ使用量を推定(MB単位)"""
        file_size_mb = model_file_path.stat().st_size / (1024 * 1024)
        return file_size_mb * multiplier


class TransformersLoader(BaseModelLoader):
    """Transformersモデルのローダー"""

    def load_components(self, model_path: str) -> dict[str, Any] | None:
        """Transformersモデルをロード"""
        # --- メモリ制約チェック ---
        model_size_mb = self.get_model_size()
        if model_size_mb > 0:  # サイズが不明(0)の場合はチェックをスキップ
            available_memory_bytes = psutil.virtual_memory().available
            required_memory_bytes = model_size_mb * 1024 * 1024
            available_memory_gb = available_memory_bytes / (1024**3)
            required_memory_gb = required_memory_bytes / (1024**3)
            logger.debug(
                f"メモリチェック ({self.model_name}): 必要={required_memory_gb:.3f}GB, 利用可能={available_memory_gb:.3f}GB"
            )
            if available_memory_bytes < required_memory_bytes:
                error_detail = (
                    f"モデル '{self.model_name}' ({required_memory_gb:.3f}GB) のロードに失敗しました。"
                    f"利用可能なシステムメモリ ({available_memory_gb:.3f}GB) が不足しています。"
                )
                error_msg = f"メモリ不足エラー: {error_detail}"
                logger.error(error_msg)
                raise OutOfMemoryError(error_detail)
        # --- メモリ制約チェック完了 ---

        if self.model_name in self._MODEL_STATES:
            logger.debug(f"モデル '{self.model_name}' は既にロード済み")
            return None

        try:
            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModelForVision2Seq.from_pretrained(model_path).to(self.device)
            components = {"model": model, "processor": processor}

            if self.model_name not in self._MODEL_SIZES:
                size = self._calculate_transformer_size(model)
                self._MODEL_SIZES[self.model_name] = size

            self._MODEL_STATES[self.model_name] = f"on_{self.device}"
            return components

        except (torch.cuda.OutOfMemoryError, MemoryError, OSError) as e:
            error_detail = f"モデル '{self.model_name}' のロード中にメモリ不足が発生しました (デバイス: {self.device})。詳細: {e}"
            error_msg = f"メモリ不足エラー: {error_detail}"
            logger.error(error_msg)
            if isinstance(e, torch.cuda.OutOfMemoryError) and self.device.startswith("cuda"):
                try:
                    logger.error(torch.cuda.memory_summary(device=self.device))
                except Exception as mem_e:
                    logger.error(f"CUDAメモリ情報取得失敗: {mem_e}")
            raise OutOfMemoryError(error_detail) from e

    def _calculate_transformer_size(self, model: torch.nn.Module) -> float:
        """Transformerモデルのメモリ使用量を計算(MB単位)"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 * 1024)
        return param_size + buffer_size


class TransformersPipelineLoader(BaseModelLoader):
    """TransformersPipelineモデルのローダー"""

    def __init__(self, model_name: str, device: str):
        super().__init__(model_name, device)

    def load_components(self, task: str, model_path: str, batch_size: int) -> dict[str, Any] | None:
        """Pipelineモデルをロード"""
        if self.model_name in self._MODEL_STATES:
            logger.debug(f"モデル '{self.model_name}' は既にロード済み")
            return None

        try:
            pipeline_obj = pipeline(
                task,
                model_path,
                device=self.device,
                batch_size=batch_size,
                use_fast=True,
            )
            components = {"pipeline": pipeline_obj}

            if self.model_name not in self._MODEL_SIZES:
                size = self._calculate_transformer_size(pipeline_obj.model)
                self._MODEL_SIZES[self.model_name] = size

            self._MODEL_STATES[self.model_name] = f"on_{self.device}"
            return components

        except (torch.cuda.OutOfMemoryError, MemoryError, OSError) as e:
            error_detail = f"モデル '{self.model_name}' (Pipeline) のロード中にメモリ不足が発生しました (デバイス: {self.device})。詳細: {e}"
            error_msg = f"メモリ不足エラー: {error_detail}"
            logger.error(error_msg)
            if isinstance(e, torch.cuda.OutOfMemoryError) and self.device.startswith("cuda"):
                 try:
                     logger.error(torch.cuda.memory_summary(device=self.device))
                 except Exception as mem_e:
                     logger.error(f"CUDAメモリ情報取得失敗: {mem_e}")
            raise OutOfMemoryError(error_detail) from e

    def _calculate_transformer_size(self, model: torch.nn.Module) -> float:
        """Transformerモデルのメモリ使用量を計算(MB単位)"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 * 1024)
        return param_size + buffer_size


class ONNXLoader(BaseModelLoader):
    """ONNXモデルのローダー"""

    def load_components(self, model_path: str) -> dict[str, Any]:
        """ONNXモデルをロード"""
        try:
            csv_path, model_repo_or_path = utils.download_onnx_tagger_model(model_path)
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers()
                else ["CPUExecutionProvider"]
            )

            logger.debug(f"ONNXモデルをロード中: '{model_repo_or_path}'")
            session = ort.InferenceSession(str(model_repo_or_path), providers=providers)
            components = {"session": session, "csv_path": csv_path}

            if self.model_name not in self._MODEL_SIZES:
                size = self._calculate_model_size(Path(model_repo_or_path), 1.5)
                self._MODEL_SIZES[self.model_name] = size

            return components

        except (MemoryError, OSError, Exception) as e: # MemoryError, OSError を追加し、汎用 Exception を最後に
            # メモリ関連のエラーか判定
            is_memory_error = False
            if isinstance(e, (MemoryError, OSError)):
                is_memory_error = True
            elif isinstance(e, Exception) and ("Failed to allocate memory" in str(e) or "CUDA error" in str(e)):
                 is_memory_error = True

            if is_memory_error:
                error_detail = f"モデル '{self.model_name}' (ONNX) のロード中にメモリ不足が発生しました。詳細: {e}"
                error_msg = f"メモリ不足エラー: {error_detail}"
                logger.error(error_msg)
                raise OutOfMemoryError(error_detail) from e
            else: # メモリ関連以外の予期せぬエラー
                logger.error(f"ONNXモデルロード中に予期せぬエラー: {e}")
                raise


class TensorFlowLoader(BaseModelLoader):
    """TensorFlowモデルのローダー"""

    def load_components(self, model_path: str, model_format: str) -> dict[str, Any]:
        """TensorFlowモデルをロード"""
        try:
            model_dir = utils.load_file(model_path)
            components: dict[str, Any] = {"model_dir": model_dir}

            if model_format == "h5":
                h5_path = next(model_dir.glob("*.h5"))
                if not h5_path:
                    raise FileNotFoundError(f"H5ファイルが見つかりません: {model_dir}")
                logger.info(f"H5モデルをロード中: {h5_path}")
                components["model"] = tf.keras.models.load_model(h5_path, compile=True)
                multiplier = 1.2
                target_path = h5_path

            elif model_format == "saved_model":
                logger.info(f"SavedModelをロード中: {model_dir}")
                components["model"] = tf.saved_model.load(model_dir)
                multiplier = 1.3
                target_path = model_dir

            else:  # pb
                pb_path = next(model_dir.glob("*.pb"))
                if not pb_path:
                    raise FileNotFoundError(f"PBファイルが見つかりません: {model_dir}")
                logger.info(f"PBモデルをロード中: {pb_path}")
                components["model"] = tf.saved_model.load(model_dir)
                multiplier = 1.3
                target_path = pb_path

            if self.model_name not in self._MODEL_SIZES:
                size = self._calculate_model_size(target_path, multiplier)
                self._MODEL_SIZES[self.model_name] = size

            return components

        except Exception as e:
            logger.error(f"TensorFlowモデルのロードに失敗: '{self.model_name}'\n{e}")
            raise


class CLIPLoader(BaseModelLoader):
    """CLIPモデルのローダー"""

    def load_components(
        self,
        base_model: str,
        model_path: str,
        activation_type: str | None = None,
        final_activation_type: str | None = None,
    ) -> dict[str, Any] | None:
        """CLIPモデルをロード"""
        if self.model_name in self._MODEL_STATES:
            self.logger.debug(f"モデル '{self.model_name}' は既に読み込まれています。")
            return None

        logger.info(f"モデル '{self.model_name}' のロードを開始します...")
        model_dict = create_clip_model(
            base_model=base_model,
            model_path=model_path,
            device=self.device,
            activation_type=activation_type,
            final_activation_type=final_activation_type,
        )

        if self.model_name not in self._MODEL_SIZES:
            size = self._calculate_transformer_size(model_dict["model"])
            self._MODEL_SIZES[self.model_name] = size

        self._MODEL_STATES[self.model_name] = f"on_{self.device}"
        logger.info(f"モデル '{self.model_name}' のロードが完了しました")
        return model_dict

    def _calculate_transformer_size(self, model: torch.nn.Module) -> float:
        """Transformerモデルのメモリ使用量を計算(MB単位)"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 * 1024)
        return param_size + buffer_size


class ModelLoad:
    """
    ModelLoadクラスの実装

    このモジュールは、モデルのロード、メモリ管理、およびキャッシュ制御を担当します。
    二階層のローダー構造により、各モデルタイプの特性に応じた効率的なモデル管理を実現します。
    """

    _MODEL_STATES: ClassVar[dict[str, str]] = {}
    _MEMORY_USAGE: ClassVar[dict[str, float]] = {}
    _MODEL_LAST_USED: ClassVar[dict[str, float]] = {}
    _CACHE_RATIO: ClassVar[float] = 0.5
    _MODEL_SIZES: ClassVar[dict[str, float]] = {}
    logger = logging.getLogger(__name__)

    @staticmethod
    def get_model_size(model_name: str) -> float:
        """モデルの推定メモリ使用量を取得(MB単位)"""
        base_loader = BaseModelLoader(model_name, "cpu")
        return base_loader.get_model_size()

    @staticmethod
    def get_max_cache_size() -> float:
        """システムの最大メモリに基づいてキャッシュサイズを計算"""
        base_loader = BaseModelLoader("", "cpu")
        return base_loader.get_max_cache_size()

    @staticmethod
    def _clear_cache_if_needed(model_name: str, model_size: float) -> None:
        """必要に応じて古いモデルをキャッシュから削除"""
        base_loader = BaseModelLoader(model_name, "cpu")
        base_loader._clear_cache_if_needed(model_size)

    @staticmethod
    def cache_to_main_memory(model_name: str, components: dict[str, Any]) -> dict[str, Any]:
        """メモリ管理を行いながらモデルをキャッシュ"""
        if model_name in ModelLoad._MODEL_STATES and ModelLoad._MODEL_STATES[model_name] == "on_cpu":
            ModelLoad.logger.debug(f"モデル '{model_name}' は既にCPUにあります。")
            ModelLoad._MODEL_LAST_USED[model_name] = time.time()
            return components

        model_size = ModelLoad.get_model_size(model_name)
        model_size_gb = model_size / 1024
        ModelLoad.logger.info(f"モデル '{model_name}' の推定サイズ: {model_size_gb:.3f}GB")

        ModelLoad._clear_cache_if_needed(model_name, model_size)

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
                f"(サイズ: {model_size_gb:.3f}GB, "
                f"現在のキャッシュ使用量: {sum(ModelLoad._MEMORY_USAGE.values()):.1f}MB/{max_cache:.1f}MB)"
            )

            return components

        except Exception as e:
            ModelLoad.logger.error(f"モデルのキャッシュに失敗しました: {e!s}")
            return components

    @staticmethod
    def restore_model_to_cuda(model_name: str, device: str, components: dict[str, Any]) -> dict[str, Any]:
        """モデルをCUDAデバイスに復元"""
        if model_name not in ModelLoad._MODEL_STATES:
            ModelLoad.logger.warning(f"モデル '{model_name}' の状態が不明です。")
            return components

        if ModelLoad._MODEL_STATES[model_name] == f"on_{device}":
            ModelLoad.logger.debug(f"モデル '{model_name}' は既に {device} にあります。")
            return components

        try:
            for component_name, component in components.items():
                if component_name == "pipeline":
                    if hasattr(component, "model"):
                        component.model.to(device)
                elif hasattr(component, "to"):
                    component.to(device)

            ModelLoad._MODEL_STATES[model_name] = f"on_{device}"
            ModelLoad._MODEL_LAST_USED[model_name] = time.time()

            return components

        except (torch.cuda.OutOfMemoryError, MemoryError, OSError) as e:
            error_detail = f"モデル '{model_name}' の CUDA デバイス '{device}' への復元中にメモリ不足が発生しました。詳細: {e}"
            error_msg = f"メモリ不足エラー: {error_detail}"
            ModelLoad.logger.error(error_msg)
            if isinstance(e, torch.cuda.OutOfMemoryError) and device.startswith("cuda") and torch.cuda.is_available():
                try:
                    ModelLoad.logger.error(torch.cuda.memory_summary(device=device))
                except Exception as mem_e:
                    ModelLoad.logger.error(f"CUDAメモリサマリーの取得に失敗: {mem_e}")
            raise OutOfMemoryError(error_detail) from e

    @staticmethod
    def release_model(model_name: str) -> None:
        """モデルの状態とメモリ使用量の記録を削除"""
        base_loader = BaseModelLoader(model_name, "cpu")
        base_loader.release_model(model_name)

    @staticmethod
    def release_model_components(model_name: str, components: dict[str, Any]) -> dict[str, Any]:
        """モデルのコンポーネントを解放"""
        try:
            for component_name, component in components.items():
                if component_name == "pipeline":
                    if hasattr(component, "model"):
                        del component.model
                elif hasattr(component, "cpu"):
                    component.cpu()
                    if hasattr(component, "to"):
                        component.to("cpu")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            ModelLoad.release_model(model_name)
            return components

        except Exception as e:
            ModelLoad.logger.error(f"モデルの解放に失敗しました: {e!s}")
            return components

    @staticmethod
    def load_transformers_components(
        model_name: str, model_path: str, device: str
    ) -> dict[str, Any] | None:
        """Transformersモデルをロード"""
        loader = TransformersLoader(model_name, device)
        return loader.load_components(model_path)

    @staticmethod
    def load_transformers_pipeline_components(
        task: str, model_name: str, model_path: str, device: str, batch_size: int
    ) -> dict[str, Any] | None:
        """TransformersPipelineモデルをロード"""
        loader = TransformersPipelineLoader(model_name, device)
        return loader.load_components(task, model_path, batch_size)

    @staticmethod
    def load_onnx_components(model_name: str, model_path: str, device: str) -> dict[str, Any]:
        """ONNXモデルをロード"""
        loader = ONNXLoader(model_name, device)
        return loader.load_components(model_path)

    @staticmethod
    def load_tensorflow_components(
        model_name: str,
        model_path: str,
        device: str,
        model_format: str,
    ) -> dict[str, Any]:
        """TensorFlowモデルをロード"""
        loader = TensorFlowLoader(model_name, device)
        return loader.load_components(model_path, model_format)

    @staticmethod
    def load_clip_components(
        model_name: str,
        base_model: str,
        model_path: str,
        device: str,
        activation_type: str | None = None,
        final_activation_type: str | None = None,
    ) -> dict[str, Any] | None:
        """CLIPモデルをロード"""
        loader = CLIPLoader(model_name, device)
        return loader.load_components(base_model, model_path, activation_type, final_activation_type)


class Classifier(nn.Module):
    """画像特徴量を入力として、分類スコアを出力する柔軟な分類器。

    Args:
        input_size (int): 入力特徴量の次元数
        hidden_sizes (list[int], optional): 各隠れ層のユニット数のリスト
        output_size (int, optional): 出力層のユニット数 (通常は1)
        dropout_rates (list[float], optional): 各隠れ層のドロップアウト率
        use_activation (bool, optional): 活性化関数を使用するかどうか
        activation (Type[nn.Module], optional): 使用する活性化関数
        use_final_activation (bool, optional): 最終層に活性化関数を使用するかどうか
        final_activation (Type[nn.Module], optional): 最終層に使用する活性化関数
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] | None = None,
        output_size: int = 1,
        dropout_rates: list[float] | None = None,
        use_activation: bool = False,
        activation: type[nn.Module] = nn.ReLU,
        use_final_activation: bool = False,
        final_activation: type[nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()

        # デフォルト値の設定
        if hidden_sizes is None:
            hidden_sizes = [1024, 128, 64, 16]

        if dropout_rates is None:
            dropout_rates = [0.2, 0.2, 0.1, 0.0]

        # ドロップアウト率のリストの長さを調整
        if len(dropout_rates) < len(hidden_sizes):
            dropout_rates = dropout_rates + [0.0] * (len(hidden_sizes) - len(dropout_rates))

        # レイヤーの構築
        layers: list[nn.Module] = []
        prev_size = input_size

        for _, (size, drop) in enumerate(zip(hidden_sizes, dropout_rates, strict=False)):
            layers.append(nn.Linear(prev_size, size))

            if use_activation:
                layers.append(activation())

            if drop > 0:
                layers.append(nn.Dropout(drop))

            prev_size = size

        # 出力層
        layers.append(nn.Linear(prev_size, output_size))

        # 最終活性化関数
        if use_final_activation:
            layers.append(final_activation())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ネットワークの順伝播を実行します。

        Args:
            x (torch.Tensor): 入力テンソル

        Returns:
            torch.Tensor: 処理された出力テンソル
        """
        return self.layers(x)  # type: ignore


def create_clip_model(
    base_model: str,
    model_path: str,
    device: str,
    activation_type: str | None = None,
    final_activation_type: str | None = None,
) -> dict[str, Any]:
    """どの CLIP モデルでも使用可能なモデルを作成します。

    Args:
        base_model (str): CLIP モデルの名前またはパス
        model_path (str): モデルの重みファイルのパス
        device (str): モデルを実行するデバイス ("cuda" または "cpu")
        activation_type (str): 活性化関数のタイプ ("ReLU", "GELU", "Sigmoid", "Tanh")
        final_activation_type (str): 最終層の活性化関数のタイプ ("ReLU", "GELU", "Sigmoid", "Tanh")

    Returns:
        dict[str, Any]: {
            "model": Classifier モデルインスタンス,
            "processor": CLIP プロセッサインスタンス,
            "clip_model": CLIP モデルインスタンス
        }
    """
    # 共通の CLIP モデルとプロセッサを初期化
    clip_processor = CLIPProcessor.from_pretrained(base_model)
    clip_model = CLIPModel.from_pretrained(base_model).to(device).eval()

    # 入力サイズを自動検出
    input_size = clip_model.config.projection_dim
    logger.debug(f"CLIP モデル {base_model} の特徴量次元: {input_size}")

    # モデルの重みをロード
    local_path = utils.load_file(model_path)
    state_dict = torch.load(local_path, map_location=device)

    # state_dict の構造から正しい hidden_features を推測する
    hidden_features = []
    layer_idx = 0

    # レイヤーの重みキーを探索して構造を推測
    while True:
        weight_key = f"layers.{layer_idx}.weight"
        if weight_key not in state_dict:
            break
        weight = state_dict[weight_key]
        hidden_features.append(weight.shape[0])
        # 活性化関数やドロップアウトがあるかに応じてスキップ量を調整
        # 基本的には線形層だけを考慮
        next_idx = layer_idx + 1
        while f"layers.{next_idx}.weight" not in state_dict and next_idx < layer_idx + 5:
            next_idx += 1
        layer_idx = next_idx

    # 最後の出力層を除外 (必要な場合)
    if hidden_features and len(hidden_features) > 1:
        hidden_features = hidden_features[:-1]

    if not hidden_features:
        # 構造を推測できなかった場合はモデルタイプによってデフォルト値を設定
        logger.warning(f"CLIP モデル {base_model} の構造を推測できませんでした。デフォルト値を設定します。")
        if "large" in base_model:
            hidden_features = [1024, 128, 64, 16]
        else:
            hidden_features = [512, 128, 64, 16]  # 小さいモデル用に調整

    logger.info(f"推測された hidden_features: {hidden_features}")

    # 活性化関数の設定マップ
    activation_map = {
        "ReLU": nn.ReLU,
        "GELU": nn.GELU,
        "Sigmoid": nn.Sigmoid,
        "Tanh": nn.Tanh,
    }

    # 設定から活性化関数のパラメータを取得
    use_activation = activation_type is not None
    if use_activation and activation_type in activation_map:
        activation_func = activation_map[activation_type]
    else:
        activation_func = nn.ReLU

    use_final_activation = final_activation_type is not None
    if use_final_activation and final_activation_type in activation_map:
        final_activation_func = activation_map[final_activation_type]
    else:
        final_activation_func = nn.Sigmoid

    # モデル初期化
    logger.info("モデル初期化開始...")
    model = Classifier(
        input_size=input_size,
        hidden_sizes=hidden_features,
        output_size=1,
        dropout_rates=[0.2, 0.2, 0.1, 0.0],
        use_activation=use_activation,
        activation=activation_func,
        use_final_activation=use_final_activation,
        final_activation=final_activation_func,
    )
    logger.debug("モデル初期化完了、重みロード開始...")
    model.load_state_dict(state_dict, strict=False)
    logger.debug("重みロード完了、デバイス転送開始...")
    model = model.to(device)
    logger.debug("デバイス転送完了")

    return {"model": model, "processor": clip_processor, "clip_model": clip_model}
