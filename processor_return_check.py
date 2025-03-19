import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, TypedDict

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

logger = logging.getLogger(__name__)

config = {
    "model_path": "microsoft/git-large-coco",
    "device": "cuda",
}


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


# プロセッサの出力を表す型定義
class ProcessorOutput(TypedDict):
    pixel_values: torch.Tensor
    # 必要に応じて他のキーを追加
    # input_ids: Tensor
    # attention_mask: Tensor


class BaseTagger(ABC):
    def __init__(self, model_name: str):
        """BaseTagger を初期化します。

        Args:
            model_name (str): モデルの名前。
        """
        self.model_name = model_name
        self.config: dict[str, Any] = config
        self.model_path = self.config["model_path"]
        self.device = self.config.get("device", "cuda")

        self.components: dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def __enter__(self) -> "BaseTagger":
        pass

    @abstractmethod
    def __exit__(self, exception_type: type[Exception], exception_value: Exception, traceback: Any) -> None:
        pass

    @abstractmethod
    def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """画像リストを処理し、アノテーション結果を返します。

        Args:
            images (list[Image.Image]): 処理対象の画像リスト。

        Returns:
            list[dict]: アノテーション結果を含む辞書のリスト。
        """
        pass

    def _generate_result(self, model_output: Any, annotation_list: list[str]) -> dict[str, Any]:
        """標準化された結果の辞書を生成します。

        Args:
            model_name (str): モデルの名前。
            model_output: モデルの出力。
            annotation_list (list[str]): 各サブクラスでタグを変換したタグリスト

        Returns:
            dict: モデル出力、モデル名、タグリストを含む辞書。
        """
        return {
            "model_name": self.model_name,
            "model_output": model_output,
            "annotation": annotation_list,
        }

    @abstractmethod
    def _calculate_annotation(self, raw_output: Any) -> float:
        """モデルの生出力からタグを計算します。

        Args:
            raw_output: モデルからの生出力。

        Returns:
            float: 計算されたアノテーション。
        """
        pass

    @abstractmethod
    def _get_annotation_tag(self, annotation: float) -> list[str]:
        """計算されたアノテーションからアノテーションタグの文字列を生成します。

        Args:
            annotation (float): 計算されたアノテーション。

        Returns:
            str: 生成されたアノテーションタグ。
        """
        pass


class TransformerModel(BaseTagger):
    """Transformersライブラリを使用するモデル用の抽象クラス。
    BLIP、BLIP2、GITなどのHugging Face Transformersベースのモデルの基底クラスとして機能します。
    """

    def __init__(self, model_name: str):
        """TransformerModel を初期化します。
        Args:
            model_name (str): モデルの名前。
        """
        super().__init__(model_name)
        # 設定ファイルから追加パラメータを取得
        self.max_length = self.config.get("max_length", 75)
        self.processor_path = self.config.get("processor_path", None)

    def __enter__(self) -> "TransformerModel":
        """
        モデルの状態に基づいて、必要な場合のみロードまたは復元
        """
        loaded_model = ModelLoad.load_transformer_components(
            self.model_name,
            self.model_path,
            self.device,
        )
        if loaded_model is not None:
            self.components = loaded_model
        self.components = ModelLoad.restore_model_to_cuda(self.model_name, self.device, self.components)
        return self

    def __exit__(self, exception_type: type[Exception], exception_value: Exception, traceback: Any) -> None:
        self.components = ModelLoad.cache_to_main_memory(self.model_name, self.components)

    def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """画像からタグを予測します。"""
        results = []
        for image in images:
            try:
                processed_image = self._preprocess_image(image)
                # モデル推論
                outputs = self._run_inference(processed_image)
                # 後処理してタグに変換
                annotations = self._postprocess_output(outputs)
                # 結果を標準形式で追加
                results.append(self._generate_result(outputs, annotations))
            except ValueError as e:
                self.logger.error(f"前処理エラー: {e}")
                raise
        return results

    def _preprocess_image(self, image: Image.Image) -> ProcessorOutput:
        """画像を前処理してモデル入力形式に変換します。
        Args:
            image (Image.Image): 入力画像

        Returns:
            ProcessorOutput: モデル用に処理された入力データ（pixel_valuesなどのテンソルを含む辞書）
        """
        if self.components["processor"] is None:
            raise ValueError("画像をTensorに変換するためのProcessorが初期化されていません。")

        processor = self.components["processor"]
        # プロセッサの出力を取得してデバイスに移動
        processed_output: ProcessorOutput = processor(images=image, return_tensors="pt").to(self.device)

        print("辞書のキー:", processed_output.keys())
        for key, tensor in processed_output.items():
            print(f"キー: {key}, デバイス: {tensor.device}, 形状: {tensor.shape}")

        return processed_output

    def _run_inference(self, inputs: ProcessorOutput) -> torch.Tensor:
        """モデル推論を実行します。
        Args:
            inputs (ProcessorOutput): モデルへの入力データ

        Returns:
            torch.Tensor: モデルからの出力
        """
        model = self.components["model"]

        model_out = model.generate(**inputs)
        print(f"推論結果のデバイス: {model_out.device}, 形状: {model_out.shape}")
        return model_out

    def _postprocess_output(self, outputs: Any) -> list[str]:
        # BLIP モデルの出力後処理を実装
        processor = self.components["processor"]
        return processor.batch_decode(outputs, skip_special_tokens=True)


if __name__ == "__main__":
    with BlipTransformerModel("git-large-coco") as tagger:
        image = Image.open(Path("tests/resources/img/1_img/file01.webp"))
        results = tagger.predict([image])
        print(results)
