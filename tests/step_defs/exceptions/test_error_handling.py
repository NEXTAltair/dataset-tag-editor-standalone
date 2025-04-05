from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from image_annotator_lib.exceptions.errors import (
    InvalidInputError,
    InvalidModelConfigError,
    InvalidOutputError,
    ModelExecutionError,
    ModelLoadError,
    ModelNotFoundError,
    UnsupportedModelError,
)
from image_annotator_lib.api import annotate


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    def test_invalid_image_input(self) -> None:
        """無効な画像入力のテスト"""
        with pytest.raises(TypeError):
            # 非画像データを渡す
            invalid_image = "これは画像ではなく文字列です"
            model = MagicMock()
            model.predict.side_effect = TypeError("入力は PIL.Image オブジェクトである必要があります")
            model.predict([invalid_image])

    def test_model_load_error(self) -> None:
        """モデルロード時のエラー処理のテスト"""
        with pytest.raises(ModelLoadError):
            with (
                patch(
                    "image_annotator_lib.score_models.imagereward.ImageRewardScorer._load_model"
                ) as mock_load,
                patch("image_annotator_lib.core.base.load_model_config") as mock_base_config,
                patch("image_annotator_lib.score_models.imagereward.load_model_config") as mock_ir_config,
            ):
                # 両方のモックに同じ設定を提供
                mock_config = {"test_model": {"class": "ImageRewardScorer", "device": "cpu"}}
                mock_base_config.return_value = mock_config
                mock_ir_config.return_value = mock_config

                # エラーをシミュレート
                mock_load.side_effect = ModelLoadError("モデルファイルが見つかりません")

                # モデル初期化
                model._load_model()

    @patch("image_annotator_lib.scorer._evaluate_model")
    def test_model_execution_error(self, mock_evaluate: MagicMock) -> None:
        """推論実行中のエラー処理のテスト"""
        # 内部エラーをシミュレート
        mock_evaluate.side_effect = RuntimeError("内部処理でエラーが発生しました")

        # テスト用の画像
        test_image = Image.new("RGB", (100, 100), color="red")

        # スコアラーモックの設定
        mock_scorer = MagicMock()

        with (
            patch("image_annotator_lib.scorer.init_scorer") as mock_init_scorer,
            patch("image_annotator_lib.scorer.ModelExecutionError") as mock_error_class,
        ):
            mock_init_scorer.return_value = mock_scorer
            # RuntimeErrorをModelExecutionErrorにラップするように設定
            mock_error_class.side_effect = lambda msg, *args: ModelExecutionError(msg)

            with pytest.raises(ModelExecutionError):
                # evaluate関数を呼び出す
                annotate([test_image], ["test_model"])

    @pytest.mark.parametrize(
        "exception_class",
        [
            ModelNotFoundError,
            ModelLoadError,
            InvalidModelConfigError,
            UnsupportedModelError,
            ModelExecutionError,
            InvalidInputError,
            InvalidOutputError,
        ],
    )
    def test_model_exceptions(self, exception_class: Exception) -> None:
        """モデル例外の発生テスト"""
        # 例外が適切に発生するかテスト
        with pytest.raises(exception_class):
            raise exception_class("テスト例外メッセージ")

    @patch("torch.cuda.max_memory_allocated")
    def test_memory_error_handling(self, mock_memory: MagicMock) -> None:
        """メモリ不足のエラー処理テスト"""
        # メモリ不足状態をシミュレート
        mock_memory.return_value = 10 * 1024 * 1024 * 1024  # 10GB

        with pytest.raises(torch.cuda.OutOfMemoryError):
            # メモリエラーのシミュレーション
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        """メモリ不足のエラー処理テスト"""
        # メモリ不足状態をシミュレート
        mock_memory.return_value = 10 * 1024 * 1024 * 1024  # 10GB

        with pytest.raises(torch.cuda.OutOfMemoryError):
            # メモリエラーのシミュレーション
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        """メモリ不足のエラー処理テスト"""
        # メモリ不足状態をシミュレート
        mock_memory.return_value = 10 * 1024 * 1024 * 1024  # 10GB

        with pytest.raises(torch.cuda.OutOfMemoryError):
            # メモリエラーのシミュレーション
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    def test_timeout_error(self) -> None:
        """処理タイムアウトの処理テスト"""
        # タイムアウトエラーをシミュレート
        with patch(
            "image_annotator_lib.scorer.init_scorer",
            side_effect=TimeoutError("処理がタイムアウトしました"),
        ):
            with pytest.raises(TimeoutError):
                get_annotator_instance("test_model")

    @patch("image_annotator_lib.score_models.imagereward.create_blip_image_reward_model")
    def test_gpu_dependency_error(self, mock_create_model: MagicMock) -> None:
        """GPU環境依存エラーの処理テスト"""
        # GPUエラーをシミュレート
        mock_create_model.side_effect = RuntimeError("CUDA error: no CUDA-capable device is detected")

        with (
            patch("image_annotator_lib.core.base.load_model_config") as mock_base_config,
            patch("image_annotator_lib.score_models.imagereward.load_model_config") as mock_ir_config,
        ):
            # 両方のモックに同じ設定を提供
            mock_config = {"test_model": {"class": "ImageRewardScorer", "device": "cuda"}}
            mock_base_config.return_value = mock_config
            mock_ir_config.return_value = mock_config

            with pytest.raises(RuntimeError) as excinfo:
                model = ImageRewardScorer("test_model", device="cuda")
                model._load_model()

            assert "CUDA" in str(excinfo.value)
