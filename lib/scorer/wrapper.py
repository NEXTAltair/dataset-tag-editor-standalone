from typing import TypedDict, NotRequired, Any
import torch

from PIL.Image import Image

from userscripts.taggers.aesthetic_shadow import AestheticShadow
from userscripts.taggers.cafeai_aesthetic_classifier import CafeAIAesthetic
from userscripts.taggers.improved_aesthetic_predictor import ImprovedAestheticPredictor as ImprovedAesthetic
from userscripts.taggers.waifu_aesthetic_classifier import WaifuAesthetic


# 予測結果の型定義
class ScorerPrediction(TypedDict):
    """画像スコアリング結果を表す型

    Attributes:
        raw_output: 生推論結果（各スコアラーの実装に依存）
        formatted_tags: フォーマット済みタグ
        success: 推論成功フラグ
        error: エラーメッセージ（推論失敗時のみ）
        model: モデル識別子（存在する場合）
    """
    raw_output: Any  # 生の予測結果をそのまま保持
    formatted_tags: list[str]
    success: bool
    error: NotRequired[str]
    model: NotRequired[str]

class BatchScorerOutput(TypedDict):
    """バッチ処理用出力フォーマット

    Attributes:
        results: 単一予測結果のリスト
        batch_success: 全体の成功ステータス
        batch_error: バッチ全体のエラー
    """
    results: list[ScorerPrediction]
    batch_success: bool
    batch_error: NotRequired[str]

class ScorerWrapper:
    """画像の審美的スコアリングモデルをラップするクラス

    このクラスは、各種スコアラーの機能を統一的なインターフェースで提供します。
    コンテキストマネージャとしても機能し、リソースの適切な管理を行います。
    """

    def __init__(self, model_name: str, batch_size: int = 1):
        """ScorerWrapperを初期化します

        Args:
            model_name: 使用するモデルの名前
            batch_size: バッチ処理時のバッチサイズ（デフォルト: 1）

        Raises:
            ValueError: 無効なモデル名が指定された場合
        """
        self.model_name = model_name
        self.batch_size = batch_size
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

    def predict(self, image: Image) -> ScorerPrediction:
        """単一画像の予測を実行します

        Args:
            image: 入力画像

        Returns:
            ScorerPrediction: 予測結果
            例: {
                "raw_output": ...,  # スコアラーの実装に依存
                "formatted_tags": [...],  # スコアラーの実装に依存
                "success": True,
                "model": "waifu-aesthetic"
            }
        """
        try:
            # 生データ取得（scorer属性経由）
            raw, score_tag = self.scorer.predict(image)

            return {
                "raw_output": raw,  # 生の予測結果をそのまま保持
                "formatted_tags": score_tag,
                "success": True,
                "model": getattr(self.scorer, "name", lambda: "unknown")(),
            }
        except Exception as e:
            return {
                "raw_output": None,
                "formatted_tags": [],
                "success": False,
                "error": str(e),
                "model": getattr(self.scorer, "name", lambda: "unknown")(),
            }

    def predict_batch(self, images: list[Image]) -> BatchScorerOutput:
        """複数の画像に対してバッチ処理でスコアを予測します

        Args:
            images: 入力画像のリスト

        Returns:
            BatchScorerOutput: バッチ処理結果
            例: {
                "results": [
                    {
                        "raw_output": ...,  # スコアラーの実装に依存
                        "formatted_tags": [...],  # スコアラーの実装に依存
                        "success": True,
                        "model": "waifu-aesthetic"
                    },
                    ...
                ],
                "batch_success": True
            }

        Raises:
            AttributeError: バッチ処理に対応していないスコアラーの場合
        """
        try:
            results = []
            has_error = False
            error_message = None

            for img in images:
                try:
                    result = self.predict(img)
                    results.append(result)
                    if not result["success"]:
                        has_error = True
                        error_message = result.get("error", "Unknown error")
                except Exception as e:
                    has_error = True
                    error_message = str(e)
                    results.append({
                        "raw_output": None,
                        "formatted_tags": [],
                        "success": False,
                        "error": str(e),
                        "model": getattr(self.scorer, "name", lambda: "unknown")(),
                    })

            output: BatchScorerOutput = {
                "results": results,
                "batch_success": not has_error
            }
            if error_message:
                output["batch_error"] = error_message
            return output

        except Exception as e:
            return {
                "results": [],
                "batch_success": False,
                "batch_error": str(e)
            }

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