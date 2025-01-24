from unittest.mock import patch, MagicMock
import pytest
from PIL import Image

from lib.scorer.wrapper import ScorerWrapper


class TestScorerWrapper:
    """ScorerWrapperクラスのテストスイート

    このテストスイートでは、画像の審美的スコアリング機能をラップする機能をテストします。
    主に以下の機能をテストします：
    - 各種スコアラーモデルの初期化と管理
    - 画像のスコア予測機能
    - バッチ処理によるスコア予測
    - リソース管理（コンテキストマネージャ）
    """

    @pytest.fixture
    def mock_image(self):
        """テスト用の画像モックを作成するフィクスチャ

        Returns:
            MagicMock: PIL.Image.Imageのモック
        """
        return MagicMock(spec=Image.Image)

    @pytest.fixture
    def scorer(self):
        """ScorerWrapperインスタンスを作成するフィクスチャ

        Returns:
            ScorerWrapper: テスト用のScorerWrapperインスタンス
        """
        return ScorerWrapper(model_name="waifu-aesthetic")

    def test_init_with_valid_model(self):
        """有効なモデル名での初期化テスト

        期待される動作:
            - 指定したモデルで正しく初期化される
            - モデルが開始状態になる
        """
        with patch("lib.scorer.wrapper.WaifuAesthetic") as mock_scorer:
            scorer = ScorerWrapper(model_name="waifu-aesthetic")
            mock_scorer.assert_called_once()
            mock_scorer.return_value.start.assert_called_once()

    def test_get_available_models(self):
        """使用可能なモデル一覧取得テスト

        期待される動作:
            - 使用可能なスコアリングモデル名とその説明が辞書形式で取得できる
            - 全ての美的評価モデルが含まれている
            - 説明文に具体的な情報が含まれている
        """
        models = ScorerWrapper.get_available_models()

        # 基本的な型チェック
        assert isinstance(models, dict)
        assert all(isinstance(desc, str) for desc in models.values())

        # 全ての美的評価モデルが含まれていることを確認
        assert "waifu-aesthetic" in models
        assert "aesthetic-shadow" in models
        assert "cafeai-aesthetic" in models
        assert "improved-aesthetic" in models

        # 説明文に具体的な情報が含まれていることを確認
        assert any("美的評価" in desc for desc in models.values())
        assert any("waifu-aesthetic-v2" in desc for desc in models.values())
        assert any("shadowlilac" in desc for desc in models.values())

    def test_init_with_invalid_model(self):
        """無効なモデル名での初期化テスト

        期待される動作:
            - ValueError例外が発生する
            - エラーメッセージに利用可能なモデル一覧が含まれる
        """
        with pytest.raises(ValueError) as exc_info:
            ScorerWrapper(model_name="invalid-model")

        error_message = str(exc_info.value)
        assert "Unknown model name: invalid-model" in error_message
        assert "Available models:" in error_message
        assert "waifu-aesthetic" in error_message

    def test_predict_single_image(self, scorer, mock_image):
        """単一画像のスコア予測テスト

        期待される動作:
            - 画像が正しくスコアリングされる
            - スコアが適切な形式で返される
            - 画像IDが含まれている
            - スコアが適切な範囲内である

        Args:
            scorer: ScorerWrapperインスタンス（フィクスチャ）
            mock_image: モック化された画像（フィクスチャ）
        """
        expected_result = {
            "image_id": "image_0",
            "model_name": "Test Scorer",
            "score": 7.0
        }

        with patch.object(scorer.scorer, "predict") as mock_predict:
            mock_predict.return_value = expected_result
            result = scorer.predict(mock_image)

            assert isinstance(result, dict)
            assert set(result.keys()) == {"image_id", "model_name", "score"}
            assert isinstance(result["score"], float)
            assert 0.0 <= result["score"] <= 10.0
            assert result["image_id"] == "image_0"
            mock_predict.assert_called_once_with(mock_image)

    def test_predict_batch(self, scorer):
        """バッチ処理でのスコア予測テスト

        期待される動作:
            - 複数画像が一括でスコアリングされる
            - 各画像のスコアが適切な形式で返される
            - 画像IDが正しく含まれている
            - スコアが適切な範囲内である

        Args:
            scorer: ScorerWrapperインスタンス（フィクスチャ）
        """
        mock_images = [MagicMock(spec=Image.Image) for _ in range(3)]
        expected_scores = [
            {"image_id": f"image_{i}", "model_name": "Test Scorer", "score": float(i + 6)}
            for i in range(3)
        ]

        with patch.object(scorer.scorer, "predict_pipe") as mock_predict_pipe:
            mock_predict_pipe.return_value = iter(expected_scores)
            results = list(scorer.predict_batch(mock_images))

            assert len(results) == len(mock_images)
            for i, result in enumerate(results):
                assert isinstance(result, dict)
                assert set(result.keys()) == {"image_id", "model_name", "score"}
                assert result["image_id"] == f"image_{i}"
                assert isinstance(result["score"], float)
                assert 0.0 <= result["score"] <= 10.0

    def test_context_manager(self, scorer):
        """コンテキストマネージャ機能のテスト

        期待される動作:
            - with文でリソースが適切に管理される
            - 終了時にstopが呼ばれる

        Args:
            scorer: ScorerWrapperインスタンス（フィクスチャ）
        """
        with patch.object(scorer.scorer, "stop") as mock_stop:
            with scorer:
                pass
            mock_stop.assert_called_once()

    def test_get_model_name(self, scorer):
        """モデル名取得機能のテスト

        期待される動作:
            - UIに表示するためのモデル名が取得できる

        Args:
            scorer: ScorerWrapperインスタンス（フィクスチャ）
        """
        with patch.object(scorer.scorer, "name", return_value="Test Scorer"):
            assert scorer.get_model_name() == "Test Scorer"