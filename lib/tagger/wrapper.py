from typing import Any, Generator, Optional

from PIL import Image

from scripts.tagger import Tagger
from scripts.dataset_tag_editor.taggers_builtin import (
    BLIP,
    BLIP2,
    Z3D_E621,
    DeepDanbooru,
    GITLarge,
    WaifuDiffusion,
    WaifuDiffusionTimm,
)


class TaggerWrapper:
    """scripts/dataset_tag_editor/taggers_builtin.pyのTaggerクラスをラップするクラス

    このクラスは、各種タガーの機能を統一的なインターフェースで提供します。
    コンテキストマネージャとしても機能し、リソースの適切な管理を行います。
    """

    def __init__(self, model_name: str, threshold: float = 0.5, batch_size: int = 4):
        """TaggerWrapperを初期化します

        Args:
            model_name: 使用するモデルの名前
            threshold: タグのフィルタリングに使用するスレッショルド値（デフォルト: 0.5）
            batch_size: バッチ処理時のバッチサイズ（デフォルト: 4）

        Raises:
            ValueError: 無効なモデル名が指定された場合
        """
        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self.tagger = self._create_tagger()
        self.tagger.start()

    def _create_tagger(self) -> Tagger:
        """指定されたモデル名に基づいてTaggerインスタンスを作成します

        Returns:
            Tagger: 作成されたTaggerインスタンス

        Raises:
            ValueError: 無効なモデル名が指定された場合
        """
        if self.model_name.startswith("waifu-diffusion"):
            if "timm" in self.model_name:
                return WaifuDiffusionTimm(
                    self.model_name,
                    threshold=self.threshold,
                    batch_size=self.batch_size,
                )
            return WaifuDiffusion(self.model_name, threshold=self.threshold)
        elif self.model_name == "deep-danbooru":
            return DeepDanbooru()
        elif self.model_name == "blip":
            return BLIP()
        elif self.model_name.startswith("blip2-"):
            return BLIP2(self.model_name)
        elif self.model_name == "git-large":
            return GITLarge()
        elif self.model_name == "z3d-e621":
            return Z3D_E621()
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def predict(
        self, image: Image.Image, threshold: Optional[float] = None
    ) -> list[str]:
        """画像からタグを予測します

        Args:
            image: 入力画像
            threshold: タグのフィルタリングに使用するスレッショルド値（オプション）
                      指定しない場合は初期化時の値が使用されます

        Returns:
            list[str]: 予測されたタグのリスト
        """
        return self.tagger.predict(image, threshold=threshold or self.threshold)

    def predict_batch(
        self, images: list[Image.Image]
    ) -> Generator[list[str], Any, None]:
        """複数の画像に対してバッチ処理でタグを予測します

        Args:
            images: 入力画像のリスト

        Returns:
            Generator[list[str], Any, None]: 各画像の予測タグのジェネレータ

        Raises:
            AttributeError: バッチ処理に対応していないタガーの場合
        """
        if not hasattr(self.tagger, "predict_pipe"):
            raise AttributeError(
                f"Tagger {self.model_name} does not support batch processing"
            )
        return self.tagger.predict_pipe(images, threshold=self.threshold)

    def get_model_name(self) -> str:
        """UIに表示するモデル名を取得します

        Returns:
            str: モデルの表示名
        """
        return self.tagger.name()

    def __enter__(self):
        """コンテキストマネージャのエントリーポイント

        Returns:
            TaggerWrapper: self
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャの終了処理

        タガーのリソースを解放します。
        """
        self.tagger.stop()
