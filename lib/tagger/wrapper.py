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
            available_models = self.get_available_models()
            raise ValueError(
                f"Unknown model name: {self.model_name}\n"
                f"Available models:\n"
                + "\n".join(f"- {k}: {v}" for k, v in available_models.items())
                + "\n\n注意: 画像の美的評価には ScorerWrapper を使用してください"
            )

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

    @staticmethod
    def get_available_models() -> dict[str, str]:
        """使用可能なモデル名とその説明を取得します

        Returns:
            dict[str, str]: キーがモデル名のパターン、値がその説明の辞書

        Note:
            モデルは以下の2つのカテゴリに分類されます：
            1. タグ付けモデル: アニメ画像やDanbooru形式のタグ付け
            2. 説明文生成モデル: 一般的な画像説明文の生成

            モジュールから動的に利用可能なモデルを取得します。
        """
        from importlib import import_module
        import inspect
        from pathlib import Path

        models = {}

        # タグ付けモデルの情報を取得
        def get_model_info(module_path: str) -> None:
            try:
                module = import_module(module_path)
                for name, cls in inspect.getmembers(module, inspect.isclass):
                    if hasattr(cls, "get_model_info"):
                        model_info = cls.get_model_info()
                        if model_info:
                            models.update(model_info)
            except ImportError:
                pass

        # scripts/dataset_tag_editor/interrogatorsからモデルを取得
        interrogators_dir = Path("scripts/dataset_tag_editor/interrogators")
        if interrogators_dir.exists():
            for file in interrogators_dir.glob("*.py"):
                if file.name != "__init__.py":
                    module_name = f"scripts.dataset_tag_editor.interrogators.{file.stem}"
                    get_model_info(module_name)

        # デフォルトのモデル情報（モジュールから取得できない場合のフォールバック）
        default_models = {
            "waifu-diffusion-*": "アニメ画像向けタグ生成 (例: waifu-diffusion-v1-4) - SmilingWolfのモデルを使用",
            "waifu-diffusion-*-timm": "アニメ画像向けタグ生成 (timmバージョン) - バッチ処理対応",
            "deep-danbooru": "Danbooru形式のタグ付け - DeepDanbooruモデルを使用",
            "z3d-e621": "E621形式のタグ付け - Z3D-E621-Convnextモデルを使用",
            "blip": "一般的な画像説明文生成 - BLIP-Large-Captioningモデルを使用",
            "blip2-*": "高性能な画像説明文生成 (例: blip2-opt-2.7b) - Salesforceのモデルを使用",
            "git-large": "詳細な画像説明文生成 - GIT-Large-COCOモデルを使用"
        }

        # モジュールから取得できなかったモデルはデフォルト情報を使用
        for model_id, description in default_models.items():
            if model_id not in models:
                models[model_id] = description

        return models
