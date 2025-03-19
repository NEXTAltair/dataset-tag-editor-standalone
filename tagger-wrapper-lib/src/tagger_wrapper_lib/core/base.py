import logging
from abc import ABC, abstractmethod
from typing import Any, TypedDict

import numpy as np
import torch
from PIL import Image

from .model_factory import ModelLoad
from .utils import load_model_config

logger = logging.getLogger(__name__)


# 前処理された画像データを表す型定義
class ProcessorOutput(TypedDict):
    pixel_values: torch.Tensor


class BaseTagger(ABC):
    def __init__(self, model_name: str):
        """BaseTagger を初期化します。

        Args:
            model_name (str): モデルの名前。
        """
        self.model_name = model_name
        self.config: dict[str, Any] = load_model_config()[model_name]

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
        self.processor_path = self.config.get("processor_path", self.model_path)

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
                token_ids = self._run_inference(processed_image)
                # 後処理してタグに変換
                annotations = self._postprocess_output(token_ids)
                # 結果を標準形式で追加
                results.append(self._generate_result(token_ids, annotations))
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

        self.logger.debug(f"辞書のキー: {processed_output.keys()}")
        for key, tensor in processed_output.items():
            self.logger.debug(f"キー: {key}, デバイス: {tensor.device}, 形状: {tensor.shape}")

        return processed_output

    def _run_inference(self, processed_image: ProcessorOutput) -> torch.Tensor:
        """モデル推論を実行します。
        Args:
            processed_image (ProcessorOutput): モデルへの入力データ

        Returns:
            torch.Tensor: モデルからの出力
        """
        model = self.components["model"]

        model_out: torch.Tensor = model.generate(**processed_image)
        self.logger.debug(f"推論結果のデバイス: {model_out.device}, 形状: {model_out.shape}")
        return model_out

    def _postprocess_output(self, token_ids: torch.Tensor) -> list[str]:
        # BLIP モデルの出力後処理を実装
        processor = self.components["processor"]
        annotations_list: list[str] = processor.batch_decode(token_ids, skip_special_tokens=True)
        return annotations_list


    class ONNXModelTagger(BaseTagger):
        """ONNXランタイムを使用するモデル用の抽象クラス。"""
        def __init__(self, model_name: str):
            """ONNXModelTagger を初期化します。
            Args:
                model_name (str): モデルの名前。
            """
            super().__init__(model_name=model_name)
            # 設定ファイルから追加パラメータを取得
            self.annotations_path = self.config["annotations_path"]
            self.threshold = self.config.get("threshold", 0.5)
            self.input_size = self.config.get("input_size", (448, 448))
            self.labels = []  # __enter__でロード

        def __enter__(self) -> "ONNXModelTagger":
            """
            モデルの状態に基づいて、必要な場合のみロードまたは復元
            """
            loaded_model = ModelLoad.load_onnx_components(
                self.model_name,
                self.model_path,
                self.device,
            )
            if loaded_model is not None:
                self.components = loaded_model
            self.components = ModelLoad.restore_model_to_cuda(self.model_name, self.device, self.components)
            return self

        def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
            """画像からタグを予測します。"""
            results = []
            for image in images:
                # 前処理
                input_data = self._preprocess_image(image)
                # モデル推論
                output_data = self._run_inference(input_data)
                # 後処理してタグに変換
                annotations = self._postprocess_output(output_data)
                # 結果を標準形式で追加
                results.append(self._generate_result(output_data, annotations))
            return results

        @abstractmethod
        def _preprocess_image(self, image: Image.Image) -> np.ndarray:
            """画像を前処理してONNXモデル入力形式に変換します。
            Args:
                image (Image.Image): 入力画像

            Returns:
                np.ndarray: ONNXモデル用に処理された入力データ
            """
            pass

        @abstractmethod
        def _run_inference(self, input_data: np.ndarray) -> np.ndarray:
            """ONNXモデル推論を実行します。
            Args:
                input_data (np.ndarray): モデルへの入力データ

            Returns:
                np.ndarray: モデルからの出力
            """
            pass

        @abstractmethod
        def _postprocess_output(self, output_data: np.ndarray) -> list[str]:
            """ONNXモデル出力を処理してタグのリストに変換します。
            Args:
                output_data (np.ndarray): モデルからの出力

            Returns:
                list[str]: タグのリスト
            """
            pass


class PipelineModel(BaseTagger):
    def __init__(self, model_name: str):
        """PipelineModel を初期化します。

        Args:
            model_name (str): モデルの名前。
        """
        super().__init__(model_name=model_name)
        self.batch_size = self.config.get("batch_size", 8)

    def __enter__(self) -> "PipelineModel":
        """
        モデルの状態に基づいて、必要な場合のみロードまたは復元
        """
        loaded_model = ModelLoad.pipeline_model_load(
            self.model_name,
            self.model_path,
            self.batch_size,
            self.device,
        )
        if loaded_model is not None:
            self.components = loaded_model
        self.components = ModelLoad.restore_model_to_cuda(self.model_name, self.device, self.components)
        return self

    def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """パイプラインモデルで画像リストのアノテーション結果を予測します。

        Args:
            images (list[Image.Image]): アノテーション対象の画像リスト。

        Returns:
            list[dict]: アノテーション結果の辞書リスト。
        """
        results = []
        for image in images:
            pipeline_model = self.components["pipeline"]
            raw_output = pipeline_model(image)
            self.logger.debug(f"モデル '{self.model_name}' に処理された生の出力結果: {raw_output}")
            annotation = self._calculate_annotation(raw_output)
            annotation_tag = self._get_annotation_tag(annotation)
            results.append(
                {
                    "model_name": self.model_name,
                    "model_output": raw_output,
                    "annotation_tag": annotation_tag,
                }
            )
        return results

    @abstractmethod
    def _calculate_annotation(self, raw_output: Any) -> float:
        """出力からアノテーションを計算します。

        Args:
            raw_output: モデルからの生出力。

        Returns:
            float: 計算されたアノテーション。
        """
        pass

    @abstractmethod
    def _get_annotation_tag(self, annotation_float: float) -> list[str]:
        """パイプラインモデル用のアノテーションタグを生成します。

        Args:
            annotation_float (float): 計算されたアノテーション。

        Returns:
            list[str]: 生成されたアノテーションタグ。
        """
        pass


class ClipModel(BaseTagger):
    """CLIPモデルを使用するアノテーションラークラス。

    任意のCLIPモデルと組み合わせたアノテーションリングモデルを扱います。
    activation_typeとfinal_activation_typeの設定に基づいて
    活性化関数の有無を動的に決定します。

    Args:
        model_name (str): モデルの名前
    """

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.base_model = self.config["base_model"]
        self.activation_type = self.config.get("activation_type", None)
        self.final_activation_type = self.config.get("final_activation_type", None)

    def __enter__(self) -> "ClipModel":
        """
        モデルの状態に基づいて、必要な場合のみロードまたは復元
        """
        loaded_model = ModelLoad.clip_model_load(
            self.model_name,
            self.base_model,
            self.model_path,
            self.device,
            self.activation_type,
            self.final_activation_type,
        )
        if loaded_model is not None:
            self.components = loaded_model
        self.components = ModelLoad.restore_model_to_cuda(self.model_name, self.device, self.components)
        return self

    # image_embeddings 関数 (WaifuAesthetic で使用されているものを流用)
    def image_embeddings(self, image: Image.Image) -> np.ndarray[Any, np.dtype[Any]]:
        """画像から CLIP モデルを使用して埋め込みベクトルを生成します。

        Args:
            image (Image.Image): 埋め込みを生成するための入力画像
            model (CLIPModel): 特徴抽出に使用する CLIP モデル
            processor (CLIPProcessor): 画像の前処理を行う CLIP プロセッサ

        Returns:
            np.ndarray: 正規化された画像の埋め込みベクトル
        """
        processor = self.components["processor"]
        model = self.components["clip_model"]
        inputs = processor(images=image, return_tensors="pt")["pixel_values"]
        inputs = inputs.to(model.device)
        result = model.get_image_features(pixel_values=inputs).cpu().detach().numpy()
        # 正規化された埋め込みを返す
        normalized_result: np.ndarray[Any, np.dtype[Any]] = result / np.linalg.norm(result)
        return normalized_result

    def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """CLIPモデルで画像リストのアノテーション結果を予測します。

        Args:
            images (list[Image.Image]): アノテーション対象の画像リスト。

        Returns:
            list[dict]: アノテーション結果の辞書リスト。
        """

        results = []
        for image in images:
            with torch.no_grad():  # NOTE: あると推論速度が早くなる
                # 画像の特徴量抽出
                image_embedding = self.preprocess(image)
                tensor_input = self._prepare_tensor(image_embedding)

                # モデル推論
                raw_annotation = self.components["model"](tensor_input).item()
                self.logger.debug(f"モデル '{self.model_name}' に処理された生の出力結果: {raw_annotation}")

            # アノテーション計算とタグ生成
            calculated_annotation = self._calculate_annotation(raw_annotation)
            self.logger.debug(
                f"モデル '{self.model_name}' 生の結果はTensorなのでfloatに変換: {calculated_annotation}"
            )
            annotation_list = self._get_annotation_tag(calculated_annotation)
            results.append(self._generate_result(raw_annotation, annotation_list))

        return results

    def _calculate_annotation(self, raw_annotation: torch.Tensor) -> float:
        """モデルの生出力からアノテーションを計算します。
        Args:
            raw_annotation: モデルからの生出力。

        Returns:
            annotation_float (float): 計算されたアノテーション。
        """
        return float(raw_annotation)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """画像をCLIPモデル用のテンソルに前処理します。

        Args:
            image (Image.Image): 入力画像。

        Returns:
            torch.Tensor: 前処理済みのテンソル。
        """
        return torch.tensor(self.image_embeddings(image))

    def _prepare_tensor(self, model_output: Any) -> torch.Tensor:
        """テンソルを設定されたデバイスへ移動させます。

        Args:
            model_output: 変換対象の出力。

        Returns:
            torch.Tensor: 指定デバイス上のテンソル。
        """
        if isinstance(model_output, torch.Tensor):
            tensor = model_output.float()
        else:
            tensor = torch.from_numpy(model_output).float()
        return tensor.to(self.device)
