from typing import Any, Generator

from PIL import Image

from userscripts.taggers.aesthetic_shadow import AestheticShadowV2 as AestheticShadow
from userscripts.taggers.cafeai_aesthetic_classifier import CafeAIAesthetic
from userscripts.taggers.improved_aesthetic_predictor import ImprovedAestheticPredictor as ImprovedAesthetic
from userscripts.taggers.waifu_aesthetic_classifier import WaifuAesthetic


class ScorerWrapper:
    """画像の審美的スコアリングモデルをラップするクラス

    このクラスは、各種スコアラーの機能を統一的なインターフェースで提供します。
    コンテキストマネージャとしても機能し、リソースの適切な管理を行います。
    """

    def __init__(self, model_name: str):
        """ScorerWrapperを初期化します

        Args:
            model_name: 使用するモデルの名前

        Raises:
            ValueError: 無効なモデル名が指定された場合
        """
        self.model_name = model_name
        self.scorer = self._create_scorer()
        self.scorer.start()

    def _create_scorer(self):
        """指定されたモデル名に基づいてスコアラーインスタンスを作成します

        Returns:
            Any: 作成されたスコアラーインスタンス

        Raises:
            ValueError: 無効なモデル名が指定された場合
        """
        if self.model_name == "waifu-aesthetic":
            return WaifuAesthetic()
        elif self.model_name == "aesthetic-shadow":
            return AestheticShadow()
        elif self.model_name == "cafeai-aesthetic":
            return CafeAIAesthetic()
        elif self.model_name == "improved-aesthetic":
            return ImprovedAesthetic()
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def predict(self, image: Image.Image) -> list[str]:
        """画像のスコアを予測します

        Args:
            image: 入力画像

        Returns:
            list[str]: 予測されたスコアのリスト（[WD]score_Xの形式）
        """
        return self.scorer.predict(image)

    def predict_batch(
        self, images: list[Image.Image]
    ) -> Generator[list[str], Any, None]:
        """複数の画像に対してバッチ処理でスコアを予測します

        Args:
            images: 入力画像のリスト

        Returns:
            Generator[list[str], Any, None]: 各画像の予測スコアのジェネレータ

        Raises:
            AttributeError: バッチ処理に対応していないスコアラーの場合
        """
        # predict_pipeまたはpredict_batchを使用
        if hasattr(self.scorer, "predict_batch"):
            return self.scorer.predict_batch(images)
        elif hasattr(self.scorer, "predict_pipe"):
            return self.scorer.predict_pipe(images)
        else:
            raise AttributeError(
                f"Scorer {self.model_name} does not support batch processing"
            )

    def get_model_name(self) -> str:
        """UIに表示するモデル名を取得します

        Returns:
            str: モデルの表示名
        """
        return self.scorer.name()

    def __enter__(self):
        """コンテキストマネージャのエントリーポイント

        Returns:
            ScorerWrapper: self
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャの終了処理

        スコアラーのリソースを解放します。
        """
        self.scorer.stop()
