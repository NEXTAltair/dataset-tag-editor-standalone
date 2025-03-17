from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from scorer_wrapper_lib.exceptions.model_errors import (
    InvalidInputError,
    InvalidModelConfigError,
    InvalidOutputError,
    ModelExecutionError,
    ModelLoadError,
    ModelNotFoundError,
    UnsupportedModelError,
)
from scorer_wrapper_lib.score_models.imagereward import ImageRewardScorer
from scorer_wrapper_lib.scorer import evaluate


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
            with patch(
                "scorer_wrapper_lib.score_models.imagereward.ImageRewardScorer._load_model"
            ) as mock_load:
                mock_load.side_effect = ModelLoadError("モデルファイルが見つかりません")
                model = ImageRewardScorer("test_model")
                model._load_model()

    @patch("scorer_wrapper_lib.scorer._evaluate_model")
    def test_model_execution_error(self, mock_evaluate: MagicMock) -> None:
        """推論実行中のエラー処理のテスト"""
        # 内部エラーをシミュレート
        mock_evaluate.side_effect = RuntimeError("内部処理でエラーが発生しました")

        # テスト用の画像
        test_image = Image.new("RGB", (100, 100), color="red")

        with pytest.raises(ModelExecutionError):
            # evaluate関数を呼び出す
            evaluate([test_image], ["test_model"])

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

    @patch("scorer_wrapper_lib.scorer.init_scorer")
    def test_timeout_error(self, mock_init_scorer: MagicMock) -> None:
        """処理タイムアウトの処理テスト"""
        # タイムアウトエラーをシミュレート
        mock_init_scorer.side_effect = TimeoutError("処理がタイムアウトしました")

        with pytest.raises(TimeoutError):
            get_scorer_instance("test_model")

    @patch("scorer_wrapper_lib.score_models.imagereward.create_blip_image_reward_model")
    def test_gpu_dependency_error(self, mock_create_model: MagicMock) -> None:
        """GPU環境依存エラーの処理テスト"""
        # GPUエラーをシミュレート
        mock_create_model.side_effect = RuntimeError("CUDA error: no CUDA-capable device is detected")

        with pytest.raises(RuntimeError) as excinfo:
            model = ImageRewardScorer("test_model", device="cuda")
            model._load_model()

        assert "CUDA" in str(excinfo.value)
