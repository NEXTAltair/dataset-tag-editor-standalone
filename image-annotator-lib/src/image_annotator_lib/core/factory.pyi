# Stub file for image_annotator_lib.core.factory
import logging
from pathlib import Path
from typing import Any, Optional

import torch

# Assuming Classifier is defined elsewhere or here temporarily
# Ideally, move Classifier definition to its own file/stub
class Classifier(torch.nn.Module): ...

class ModelLoad:
    """Handles loading, caching, and memory management for various model types. / 様々なモデルタイプのロード、キャッシュ、メモリ管理を扱います。"""

    _MODEL_STATES: dict[str, str]
    _MEMORY_USAGE: dict[str, float]
    _MODEL_LAST_USED: dict[str, float]
    _CACHE_RATIO: float
    _MODEL_SIZES: dict[str, float]
    logger: logging.Logger

    @staticmethod
    def get_model_size(model_name: str) -> float:
        """モデルの推定メモリ使用量を取得（MB単位）。
        Gets the estimated memory usage of a model in megabytes (MB).

        Args:
            model_name (str): The name of the model. / モデルの名前。

        Returns:
            float: Estimated memory usage in MB, or 0.0 if unknown. / 推定メモリ使用量 (MB)、不明な場合は 0.0。
        """
        ...

    @staticmethod
    def get_max_cache_size() -> float:
        """システムの最大メモリに基づいてキャッシュサイズを計算します (MB単位)。
        Calculates the cache size based on the system's maximum memory (in MB).

        Returns:
            float: The calculated maximum cache size in MB. / 計算された最大キャッシュサイズ (MB)。
        """
        ...

    @staticmethod
    def _clear_cache_if_needed(model_name: str, model_size: float) -> None:
        """必要に応じて古いモデルをキャッシュから削除します。
        Removes old models from the cache if necessary to make space.

        Args:
            model_name (str): The name of the model being loaded. / ロード中のモデルの名前。
            model_size (float): The estimated size of the model being loaded (MB). / ロード中のモデルの推定サイズ (MB)。
        """
        ...

    @staticmethod
    def cache_to_main_memory(model_name: str, components: dict[str, Any]) -> dict[str, Any]:
        """モデルコンポーネントをメインメモリ (CPU) にキャッシュします。
        Caches the model components to main memory (CPU).

        Args:
            model_name (str): The name of the model. / モデルの名前。
            components (dict[str, Any]): The dictionary containing model components. / モデルコンポーネントを含む辞書。

        Returns:
            dict[str, Any]: The components dictionary (potentially modified in place). / コンポーネント辞書 (インプレースで変更される可能性あり)。
        """
        ...

    @staticmethod
    def _calculate_and_save_model_size(model_name: str, model_size: float) -> None:
        """モデルサイズを計算し、内部キャッシュと設定ファイルに保存します。
        Calculates and saves the model size to the internal cache and configuration file.

        Args:
            model_name (str): The name of the model. / モデルの名前。
            model_size (float): The calculated model size in MB. / 計算されたモデルサイズ (MB)。
        """
        ...

    @staticmethod
    def load_transformer_components(
        model_name: str, model_path: str, device: str
    ) -> Optional[dict[str, Any]]:
        """Transformer モデルのコンポーネント (モデル、プロセッサ) をロードします。
        Loads Transformer model components (model, processor).

        Args:
            model_name (str): The name of the model. / モデルの名前。
            model_path (str): Path or identifier for the model. / モデルのパスまたは識別子。
            device (str): The device to load the model onto ("cuda" or "cpu"). / モデルをロードするデバイス ("cuda" または "cpu")。

        Returns:
            Optional[dict[str, Any]]: Dictionary with "model" and "processor", or None if already loaded. / "model" と "processor" を含む辞書、またはロード済みの場合は None。
        """
        ...

    @staticmethod
    def load_onnx_components(model_name: str, model_repo: str, device: str) -> dict[str, Any]:
        """ONNX モデルのコンポーネント (セッション、CSVパス) をロードします。
        Loads ONNX model components (session, csv_path).

        Args:
            model_name (str): The name of the model. / モデルの名前。
            model_repo (str): Repository or path for the ONNX model. / ONNX モデルのリポジトリまたはパス。
            device (str): The device to run inference on ("cuda" or "cpu"). / 推論を実行するデバイス ("cuda" または "cpu")。

        Returns:
            dict[str, Any]: Dictionary with "session" and "csv_path". / "session" と "csv_path" を含む辞書。
        """
        ...

    @staticmethod
    def load_tensorflow_components(
        model_name: str, model_path: str, device: str, model_format: str = "h5"
    ) -> dict[str, Any]:
        """TensorFlow モデルのコンポーネント (モデル、モデルディレクトリ) をロードします。
        Loads TensorFlow model components (model, model_dir).

        Args:
            model_name (str): The name of the model. / モデルの名前。
            model_path (str): Path to the model file or directory. / モデルファイルまたはディレクトリへのパス。
            device (str): Target device (currently unused for TF). / ターゲットデバイス (現在 TF では未使用)。
            model_format (str): The format of the model ("h5", "saved_model", "pb"). / モデルのフォーマット ("h5", "saved_model", "pb")。

        Returns:
            dict[str, Any]: Dictionary with "model" and "model_dir". / "model" と "model_dir" を含む辞書。
        """
        ...

    @staticmethod
    def load_pipeline_components(
        model_name: str, model_path: str, batch_size: int, device: str
    ) -> Optional[dict[str, Any]]:
        """Hugging Face Pipeline モデルのコンポーネントをロードします。
        Loads Hugging Face Pipeline model components.

        Args:
            model_name (str): The name of the model. / モデルの名前。
            model_path (str): Path or identifier for the pipeline model. / パイプラインモデルのパスまたは識別子。
            batch_size (int): Batch size for the pipeline. / パイプラインのバッチサイズ。
            device (str): The device to load the pipeline onto ("cuda" or "cpu"). / パイプラインをロードするデバイス ("cuda" または "cpu")。

        Returns:
            Optional[dict[str, Any]]: Dictionary with "pipeline", or None if already loaded. / "pipeline" を含む辞書、またはロード済みの場合は None。
        """
        ...

    @staticmethod
    def load_clip_components(
        model_name: str,
        base_model: str,
        model_path: str,
        device: str,
        activation_type: Optional[str] = None,
        final_activation_type: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """CLIP ベースの Scorer モデルのコンポーネント (分類器、プロセッサ、CLIPモデル) をロードします。
        Loads CLIP-based Scorer model components (classifier, processor, clip_model).

        Args:
            model_name (str): The name of the scorer model. / スコアラーモデルの名前。
            base_model (str): Path or identifier for the base CLIP model. / ベースとなる CLIP モデルのパスまたは識別子。
            model_path (str): Path to the classifier head weights. / 分類器ヘッドの重みへのパス。
            device (str): The device to load models onto ("cuda" or "cpu"). / モデルをロードするデバイス ("cuda" または "cpu")。
            activation_type (Optional[str]): Type of activation for hidden layers. / 隠れ層の活性化関数のタイプ。
            final_activation_type (Optional[str]): Type of activation for the final layer. / 最終層の活性化関数のタイプ。

        Returns:
            Optional[dict[str, Any]]: Dictionary with "model", "processor", "clip_model", or None if already loaded. / "model", "processor", "clip_model" を含む辞書、またはロード済みの場合は None。
        """
        ...

    @staticmethod
    def restore_model_to_cuda(model_name: str, device: str, model: dict[str, Any]) -> dict[str, Any]:
        """キャッシュされたモデルを指定された CUDA デバイスに復元します。
        Restores a cached model to the specified CUDA device.

        Args:
            model_name (str): The name of the model. / モデルの名前。
            device (str): The target CUDA device (e.g., "cuda:0"). / ターゲット CUDA デバイス (例: "cuda:0")。
            model (dict[str, Any]): The dictionary containing model components (potentially on CPU). / モデルコンポーネントを含む辞書 (CPU 上にある可能性あり)。

        Returns:
            dict[str, Any]: The components dictionary with components moved to the target device. / コンポーネントがターゲットデバイスに移動されたコンポーネント辞書。
        """
        ...

    @staticmethod
    def release_model(model_name: str) -> None:
        """指定されたモデルをキャッシュから解放し、関連リソース (GPUメモリなど) をクリアします。
        Releases the specified model from the cache and clears associated resources (like GPU memory).

        Args:
            model_name (str): The name of the model to release. / 解放するモデルの名前。
        """
        ...

    @staticmethod
    def release_model_components(model_name: str, components: dict[str, Any]) -> dict[str, Any]:
        """モデルコンポーネント (ONNX セッションなど) のリソースを明示的に解放します。
        Explicitly releases resources held by model components (like ONNX sessions).

        Args:
            model_name (str): The name of the model. / モデルの名前。
            components (dict[str, Any]): The dictionary containing model components. / モデルコンポーネントを含む辞書。

        Returns:
            dict[str, Any]: The components dictionary with released components set to None. / 解放されたコンポーネントが None に設定されたコンポーネント辞書。
        """
        ...

    @staticmethod
    def _calculate_transformer_size(model: torch.nn.Module) -> float: ...
    @staticmethod
    def _calculate_model_size(model_file_path: Path, multiplier: float) -> float: ...

# Classifier definition might be moved to a separate stub/file later
# Classifier クラス定義は後で別のスタブ/ファイルに移動する可能性あり
# class Classifier(torch.nn.Module): ...
