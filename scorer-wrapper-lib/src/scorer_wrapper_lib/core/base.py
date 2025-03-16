import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
from PIL import Image

from .model_factory import create_model, image_embeddings
from .utils import load_model_config

logger = logging.getLogger(__name__)


class BaseScorer(ABC):
    def __init__(self, model_name: str):
        """BaseScorer を初期化します。

        Args:
            model_name (str): モデルの名前。
        """
        self.model_name = model_name
        self.config: dict[str, Any] = load_model_config()[model_name]
        self.device = self.config["device"]
        self.model: dict[str, Any] = {}
        self.model_state = "unloaded"  # 初期状態: モデル未ロード
        self.logger = logging.getLogger(__name__)

    def _load_model(self) -> None:
        """
        モデルファクトリを使用してモデルを読み込みます。

        Raises:
            Exception: モデル読み込み時に発生した例外をそのまま伝播します。
        """
        self.model = create_model(self.config)
        self.model_state = f"on_{self.device}"  # "on_cuda" または "on_cpu"

    def load_or_restore_model(self) -> None:
        """
        モデルの状態に基づいて、必要な場合のみロードまたは復元します。
        """
        if not self.is_model_loaded():
            # モデルが読み込まれていない場合
            self.logger.debug(f"モデル '{self.model_name}' をロードします")
            self._load_model()
        elif self.needs_gpu_restoration():
            # CPUにキャッシュされていて、GPUに戻す必要がある場合
            self.logger.debug(f"モデル '{self.model_name}' をGPUに復元します")
            self._restore_from_main_memory()

    def _release_model(self) -> None:
        """
        モデルを_LOADED_SCORERS から削除 メモリから解放し、GPUキャッシュをクリアします。
        """
        if self.model is not None:
            self.model_state = "unloaded"  # モデル解放
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def cache_to_main_memory(self) -> None:
        """モデルをCPUメモリにキャッシュします。

        モデルのすべてのコンポーネントをGPUからCPUメモリに移動します。
        これにより、GPU上のメモリは解放されますが、モデル自体は保持されるため、
        後で`_restore_from_main_memory()`を呼び出して再利用できます。

        主な用途:
        - 一時的にGPUメモリを解放したい場合
        - 複数のモデルを交互に使用する場合
        - モデル自体は保持したままGPUリソースを解放したい場合

        Note:
            このメソッドはモデルを破棄しません。モデルを完全に解放するには
            `release_resources()`を使用してください。
        """
        # 辞書内の各要素に対して処理
        for component_name, component in self.model.items():
            if component_name == "pipeline":
                # パイプラインの場合は内部モデルを移動
                if hasattr(component, "model"):
                    component.model.to("cpu")
                self.logger.debug(
                    f"[{self.__class__.__name__}] パイプライン '{component_name}' をCPUに移動しました"
                )
            elif hasattr(component, "to"):  # toメソッドを持つ場合のみCPUに移動
                component.to("cpu")
                self.logger.debug(
                    f"[{self.__class__.__name__}] コンポーネント '{component_name}' をCPUに移動しました"
                )

        # CUDAが利用可能な場合のみGPUメモリをクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 処理完了のログ
        self.logger.info(
            f"[{self.__class__.__name__}] '{self.model_name}' をメインメモリにキャッシュしました。"
        )
        self.model_state = "on_cpu"

    def _restore_from_main_memory(self) -> None:
        """CPUにキャッシュされたモデルを指定デバイスCUDAに復元します。

        `cache_to_main_memory()`を使用してCPUに移動したモデルを、
        設定されたデバイスself.device、CUDAに戻します。
        これにより、モデルは再び高速推論の準備が整います。

        主な用途:
        - キャッシュしておいたモデルを再度使用する前
        - CUDA上での処理が必要になった時
        """
        # 辞書内の各要素に対して処理
        for component_name, component in self.model.items():
            if component_name == "pipeline":
                # パイプラインの場合は内部モデルを移動
                if hasattr(component, "model"):
                    component.model.to(self.device)
                self.logger.debug(
                    f"[{self.__class__.__name__}] パイプライン '{component_name}' を{self.device}に移動しました"
                )
            elif hasattr(component, "to"):  # toメソッドを持つ場合のみデバイスに移動
                component.to(self.device)
                self.logger.debug(
                    f"[{self.__class__.__name__}] コンポーネント '{component_name}' を{self.device}に移動しました"
                )

        # 処理完了のログ
        self.logger.info(
            f"[{self.__class__.__name__}] '{self.model_name}' をメインメモリから復元しました。"
        )

    def release_resources(self) -> None:
        """モデルへの参照を解放し、メモリリソースを完全に解放します。

        このメソッドは`_release_model()`を呼び出してモデルへの参照を削除し、
        Pythonのガベージコレクションによってメモリを解放できるようにします。

        子クラスでは、このメソッドをオーバーライドして、モデル固有の
        追加リソース解放処理を実装できます。

        主な用途:
        - メモリ使用量を最小化する必要がある場合
        - モデルを完全に解放する場合
        - 新しいモデルをロードする前に既存のモデルを解放する場合
        """
        self._release_model()

    @abstractmethod
    def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """画像リストを処理し、評価結果を返します。

        Args:
            images (list[Image.Image]): 処理対象の画像リスト。

        Returns:
            list[dict]: 評価結果を含む辞書のリスト。
        """
        pass

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """画像をCLIPモデル用のテンソルに前処理します。

        Args:
            image (Image.Image): 入力画像。

        Returns:
            torch.Tensor: 前処理済みのテンソル。
        """
        return torch.tensor(image_embeddings(image, self.model["clip_model"], self.model["processor"]))

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

    def is_model_loaded(self) -> bool:
        """モデルがロード済みかどうかを確認します。"""
        return self.model_state != "unloaded"

    def is_on_gpu(self) -> bool:
        """モデルがGPU上にあるかどうかを確認します。"""
        return self.model_state == f"on_{self.device}" and "cuda" in self.device

    def is_on_cpu(self) -> bool:
        """モデルがCPU上にあるかどうかを確認します。"""
        return self.model_state == "on_cpu"

    def needs_gpu_restoration(self) -> bool:
        """モデルがCPUにあり、GPUに戻す必要があるかを確認します。"""
        return self.is_on_cpu() and "cuda" in self.device


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
    def _get_score_tag(self, score: float) -> str:
        """パイプラインモデル用のスコアタグを生成します。

        Args:
            score (float): 計算されたスコア。

        Returns:
            str: 生成されたスコアタグ。
        """
        pass


class ClipMlpModel(BaseScorer):
    def __init__(self, model_name: str):
        """ClipMlpModel を初期化します。

        Args:
            model_name (str): モデルの名前。
        """
        super().__init__(model_name=model_name)

    def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """CLIP+MLPモデルで画像リストの評価結果を予測します。

        Args:
            images (list[Image.Image]): 評価対象の画像リスト。

        Returns:
            list[dict]: 評価結果の辞書リスト。
        """
        if not self.is_model_loaded():
            self.load_or_restore_model()

        results = []
        for image in images:
            image_embedding = self.preprocess(image)
            tensor_input = self._prepare_tensor(image_embedding)
            raw_score = self.model["model"](tensor_input).item()
            self.logger.debug(f"モデル '{self.model_name}' に処理された生の出力結果: {raw_score}")
            calculated_score = self._calculate_score(raw_score)
            score_tag = self._get_score_tag(calculated_score)
            results.append(self._generate_result(raw_score, score_tag))
        return results

    def _calculate_score(self, raw_output: Any) -> float:
        """モデルの生出力からスコアを計算します。"""
        return float(raw_output)

    def _get_score_tag(self, score: float) -> str:
        """スコアからタグを生成します。"""
        return f"{self.config.get('score_prefix')}score_{int(score)}"


class ClipClassifierModel(BaseScorer):
    def __init__(self, model_name: str):
        """ClipClassifierModel を初期化します。

        Args:
            model_name (str): モデルの名前。
        """
        super().__init__(model_name=model_name)

    def _calculate_score(self, raw_output: Any) -> float:
        """モデルの生出力からスコアを計算します。"""
        return float(raw_output)

    def _get_score_tag(self, score: float) -> str:
        """スコアからタグを生成します。"""
        return f"{self.config.get('score_prefix')}score_{int(score)}"

    def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """CLIP+Classifierモデルで画像リストの評価結果を予測します。

        Args:
            images (list[Image.Image]): 評価対象の画像リスト。

        Returns:
            list[dict]: 評価結果の辞書リスト。
        """
        if not self.is_model_loaded():
            self.load_or_restore_model()

        results = []
        for image in images:
            image_embedding = self.preprocess(image)
            tensor_input = self._prepare_tensor(image_embedding)
            raw_score = self.model["model"](tensor_input).item()
            self.logger.debug(f"モデル '{self.model_name}' に処理された生の出力結果: {raw_score}")
            calculated_score = self._calculate_score(raw_score)
            score_tag = self._get_score_tag(calculated_score)
            results.append(self._generate_result(raw_score, score_tag))
        return results
