import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from PIL import Image

from .model_factory import ModelLoad
from .utils import load_model_config

logger = logging.getLogger(__name__)


class BaseScorer(ABC):
    # TODO: モデルのキャッシュ周りとかはmodel_factoryに移行するほうがきれい
    def __init__(self, model_name: str):
        """BaseScorer を初期化します。

        Args:
            model_name (str): モデルの名前。
        """
        self.model_name = model_name
        self.config: dict[str, Any] = load_model_config()[model_name]
        self.model_type = self.config["type"]
        self.base_model = self.config.get("base_model", None)
        self.model_path = self.config["model_path"]
        self.device = self.config.get("device", "cuda")
        self.activation_type = self.config.get("activation_type", None)
        self.final_activation_type = self.config.get("final_activation_type", None)
        self.model: dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    def __enter__(self) -> "BaseScorer":
        """
        モデルの状態に基づいて、必要な場合のみロードまたは復元
        """
        loaded_model = ModelLoad.load_model(
            self.model_name,
            self.model_type,
            self.base_model,
            self.model_path,
            self.device,
            self.activation_type,
            self.final_activation_type,
        )
        if loaded_model is not None:
            self.model = loaded_model
        self.model = ModelLoad.restore_model_to_cuda(self.model_name, self.device, self.model)
        return self

    def __exit__(self, exception_type: type[Exception], exception_value: Exception, traceback: Any) -> None:
        self.model = ModelLoad.cache_to_main_memory(self.model_name, self.model)

    @abstractmethod
    def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """画像リストを処理し、評価結果を返します。

        Args:
            images (list[Image.Image]): 処理対象の画像リスト。

        Returns:
            list[dict]: 評価結果を含む辞書のリスト。
        """
        pass

    def _generate_result(self, model_output: Any, score_tag: str) -> dict[str, Any]:
        """標準化された結果の辞書を生成します。

        Args:
            model_name (str): モデルの名前。
            model_output: モデルの出力。
            score_tag (str)-各サブクラスでスコアを変換したタグ

        Returns:
            dict: モデル出力、モデル名、スコアタグを含む辞書。
        """
        return {
            "model_name": self.model_name,
            "model_output": model_output,
            "score_tag": score_tag,
        }

    @abstractmethod
    def _calculate_score(self, raw_output: Any) -> float:
        """モデルの生出力からスコアを計算します。

        Args:
            raw_output: モデルからの生出力。

        Returns:
            float: 計算されたスコア。
        """
        pass

    @abstractmethod
    def _get_score_tag(self, score: float) -> str:
        """計算されたスコアからスコアタグの文字列を生成します。

        Args:
            score (float): 計算されたスコア。

        Returns:
            str: 生成されたスコアタグ。
        """
        pass


class PipelineModel(BaseScorer):
    def __init__(self, model_name: str):
        """PipelineModel を初期化します。

        Args:
            model_name (str): モデルの名前。
        """
        super().__init__(model_name=model_name)
        self.score_prefix = self.config.get("score_prefix", "")

    def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """パイプラインモデルで画像リストの評価結果を予測します。

        Args:
            images (list[Image.Image]): 評価対象の画像リスト。

        Returns:
            list[dict]: 評価結果の辞書リスト。
        """
        results = []
        for image in images:
            pipeline_model = self.model["pipeline"]
            raw_output = pipeline_model(image)
            self.logger.debug(f"モデル '{self.model_name}' に処理された生の出力結果: {raw_output}")
            score = self._calculate_score(raw_output)
            score_tag = self._get_score_tag(score)
            results.append(
                {
                    "model_name": self.model_name,
                    "model_output": raw_output,
                    "score_tag": score_tag,
                }
            )
        return results

    @abstractmethod
    def _calculate_score(self, raw_output: Any) -> float:
        """出力からスコアを計算します。

        Args:
            raw_output: モデルからの生出力。

        Returns:
            float: 計算されたスコア。
        """
        pass

    @abstractmethod
    def _get_score_tag(self, score_float: float) -> str:
        """パイプラインモデル用のスコアタグを生成します。

        Args:
            score_float (float): 計算されたスコア。

        Returns:
            str: 生成されたスコアタグ。
        """
        pass


class ClipModel(BaseScorer):
    """CLIPモデルを使用するスコアラークラス。

    任意のCLIPモデルと組み合わせたスコアリングモデルを扱います。
    activation_typeとfinal_activation_typeの設定に基づいて
    活性化関数の有無を動的に決定します。

    Args:
        model_name (str): モデルの名前
    """

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)

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
        processor = self.model["processor"]
        model = self.model["clip_model"]
        inputs = processor(images=image, return_tensors="pt")["pixel_values"]
        inputs = inputs.to(model.device)
        result = model.get_image_features(pixel_values=inputs).cpu().detach().numpy()
        # 正規化された埋め込みを返す
        normalized_result: np.ndarray[Any, np.dtype[Any]] = result / np.linalg.norm(result)
        return normalized_result

    def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """CLIPモデルで画像リストの評価結果を予測します。

        Args:
            images (list[Image.Image]): 評価対象の画像リスト。

        Returns:
            list[dict]: 評価結果の辞書リスト。
        """

        results = []
        for image in images:
            with torch.no_grad():  # NOTE: あると推論速度が早くなる
                # 画像の特徴量抽出
                image_embedding = self.preprocess(image)
                tensor_input = self._prepare_tensor(image_embedding)

                # モデル推論
                raw_score = self.model["model"](tensor_input).item()
                self.logger.debug(f"モデル '{self.model_name}' に処理された生の出力結果: {raw_score}")

            # スコア計算とタグ生成
            calculated_score = self._calculate_score(raw_score)
            score_tag = self._get_score_tag(calculated_score)
            results.append(self._generate_result(raw_score, score_tag))

        return results

    def _calculate_score(self, raw_score: torch.Tensor) -> float:
        """モデルの生出力からスコアを計算します。
        Args:
            raw_score: モデルからの生出力。

        Returns:
            score_float (float): 計算されたスコア。
        """
        return float(raw_score)

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

    @abstractmethod
    def _get_score_tag(self, score_float: float) -> str:
        pass
