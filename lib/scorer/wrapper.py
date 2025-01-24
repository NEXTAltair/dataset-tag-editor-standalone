from typing import Any, Generator, TypedDict

from PIL import Image

from userscripts.taggers.aesthetic_shadow import AestheticShadow
from userscripts.taggers.cafeai_aesthetic_classifier import CafeAIAesthetic
from userscripts.taggers.improved_aesthetic_predictor import ImprovedAestheticPredictor as ImprovedAesthetic
from userscripts.taggers.waifu_aesthetic_classifier import WaifuAesthetic


# 予測結果の型定義
class ImageScore(TypedDict):
    """画像スコアリング結果を表す型

    Attributes:
        image_id: 画像の識別子
        model_name: 使用したモデルの名前
        score: 予測スコア（0.0-10.0）
    """
    image_id: str
    model_name: str
    score: float


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
            available_models = self.get_available_models()
            raise ValueError(
                f"Unknown model name: {self.model_name}\n"
                f"Available models:\n"
                + "\n".join(f"- {k}: {v}" for k, v in available_models.items())
            )

    @staticmethod
    def get_available_models() -> dict[str, str]:
        """使用可能なスコアリングモデル名とその説明を取得します

        Returns:
            dict[str, str]: キーがモデル名、値がその説明の辞書

        Note:
            全てのモデルは画像の美的評価に特化しており、
            異なるアプローチや特徴を持つモデルを提供します。
            モジュールから動的に利用可能なモデルを取得します。
        """
        from importlib import import_module
        import inspect
        from pathlib import Path

        models = {}

        # スコアリングモデルの情報を取得
        def get_model_info(file_path: Path) -> None:
            try:
                # ファイル名からモジュール名を生成
                module_name = f"userscripts.taggers.{file_path.stem}"
                module = import_module(module_name)

                # モジュール内のクラスを検査
                for name, cls in inspect.getmembers(module, inspect.isclass):
                    if hasattr(cls, "get_model_info"):
                        model_info = cls.get_model_info()
                        if model_info:
                            models.update(model_info)
            except ImportError:
                pass

        # userscripts/taggersディレクトリからモデルを取得
        taggers_dir = Path("userscripts/taggers")
        if taggers_dir.exists():
            for file in taggers_dir.glob("*.py"):
                if not file.name.startswith("__"):
                    get_model_info(file)

        # デフォルトのモデル情報（モジュールから取得できない場合のフォールバック）
        default_models = {
            "waifu-aesthetic": "アニメ画像向け美的評価 - waifu-aesthetic-v2を使用",
            "aesthetic-shadow": "画像の美的評価 - shadowlilac/aesthetic-shadowモデルを使用",
            "cafeai-aesthetic": "カフェスタイルの美的評価 - cafeai/cafe_aesthetic_modelを使用",
            "improved-aesthetic": "改良版美的評価 - improved-aesthetic-v2を使用"
        }

        # モジュールから取得できなかったモデルはデフォルト情報を使用
        for model_id, description in default_models.items():
            if model_id not in models:
                models[model_id] = description

        return models

    def predict(self, image: Image.Image) -> ImageScore:
        """画像のスコアを予測します

        Args:
            image: 入力画像

        Returns:
            ImageScore: 予測結果を含む辞書
            {
                "image_id": str,    # 画像の識別子
                "model_name": str,  # モデル名
                "score": float,     # スコア（0.0-10.0）
            }
        """
        return self.scorer.predict(image)

    def predict_batch(
        self,
        images: list[Image.Image]
    ) -> Generator[ImageScore, None, None]:
        """複数の画像に対してバッチ処理でスコアを予測します

        Args:
            images: 入力画像のリスト

        Returns:
            Generator[ImageScore, None, None]:
            各画像の予測結果を含むジェネレータ。
            ImageScoreは以下の構造:
            {
                "image_id": str,    # 画像の識別子
                "model_name": str,  # モデル名
                "score": float,     # スコア（0.0-10.0）
            }

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
