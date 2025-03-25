import logging
from typing import Any, Optional

import onnxruntime as ort
import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)

from . import utils

logger = logging.getLogger(__name__)


class ModelLoad:
    _MODEL_STATES: dict[str, str] = {}
    logger = logging.getLogger(__name__)

    @staticmethod
    def load_transformer_components(
        model_name: str, model_path: str, device: str
    ) -> Optional[dict[str, Any]]:
        if model_name in ModelLoad._MODEL_STATES:
            ModelLoad.logger.debug(f"モデル '{model_name}' は既に読み込まれています。")
            return None

        # 適切なプロセッサとモデルを自動的に選択
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)

        ModelLoad._MODEL_STATES[model_name] = f"on_{device}"
        return {"model": model, "processor": processor}

    @staticmethod
    def load_onnx_components(model_name: str, model_repo: str, device: str) -> Optional[dict[str, Any]]:
        if model_name in ModelLoad._MODEL_STATES:
            ModelLoad.logger.debug(f"モデル '{model_name}' は既に読み込まれています。")
            return None
        # ONNXランタイムセッションの作成
        csv_path, model_path = utils.download_wd_tagger_model(model_repo)

        # 利用可能なプロバイダーを取得
        available_providers = ort.get_available_providers()
        ModelLoad.logger.debug(f"利用可能なプロバイダー: {available_providers}")

        # デバイスに基づいてプロバイダーを選択
        if device == "cuda" and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        ModelLoad.logger.info(f"ONNXモデル '{model_path}' をロードしています...")
        session = ort.InferenceSession(model_path, providers=providers)

        ModelLoad._MODEL_STATES[model_name] = f"on_{device}"
        return {"model": session, "csv_path": csv_path}

    @staticmethod
    def cache_to_main_memory(model_name: str, model: dict[str, Any]) -> dict[str, Any]:
        """モデルを CPU メモリにキャッシュします。

        モデルのすべてのコンポーネントを GPU から CPU メモリに移動します。
        これにより、GPU 上のメモリは解放されますが、モデル自体は保持されるため、
        後で `restore_model_to_cuda` を呼び出して再利用できます。

        主な用途:
        - モデル自体は保持したまま GPU リソースを解放したい場合

        Note:
            このメソッドはモデルを破棄しません。モデルを完全に解放するには
            `release_model` を使用してください。
        """
        if ModelLoad._MODEL_STATES[model_name] == "on_cpu":
            ModelLoad.logger.debug(f"モデル '{model_name}' は既に CPU にあります。")
            return model

        for component_name, component in model.items():
            if component_name == "pipeline":
                # パイプラインの場合は内部モデルを移動
                if hasattr(component, "model"):
                    component.model.to("cpu")
                ModelLoad.logger.debug(f"パイプライン '{component_name}' を CPU に移動しました")
            elif hasattr(component, "to"):  # to メソッドを持つ場合のみ CPU に移動
                component.to("cpu")
                ModelLoad.logger.debug(f"コンポーネント '{component_name}' を CPU に移動しました")

        ModelLoad._MODEL_STATES[model_name] = "on_cpu"
        return model

    @staticmethod
    def restore_model_to_cuda(model_name: str, device: str, model: dict[str, Any]) -> dict[str, Any]:
        """モデルを指定 CUDA に復元します。"""
        if ModelLoad._MODEL_STATES[model_name] == "on_cpu" and "cuda" in device:
            for component_name, component in model.items():
                if component_name == "pipeline":
                    if hasattr(component, "model"):
                        component.model.to("cuda")

                elif hasattr(component, "to"):
                    component.to("cuda")

                ModelLoad._MODEL_STATES[model_name] = "on_cuda"
                ModelLoad.logger.info(f"モデル '{model_name}' をメインメモリから復元しました。")
            return model

        return model

    @staticmethod
    def release_model(model_name: str) -> None:
        """
        モデルを _LOADED_SCORERS から削除 メモリから解放し、GPU キャッシュをクリアします。
        """
        if model_name in ModelLoad._MODEL_STATES:
            del ModelLoad._MODEL_STATES[model_name]

        # GPU メモリをクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ModelLoad.logger.info(f"モデル '{model_name}' を解放しました。")
