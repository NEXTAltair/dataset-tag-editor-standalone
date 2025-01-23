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

    def test_init_with_invalid_model(self):
        """無効なモデル名での初期化テスト

        期待される動作:
            - ValueError例外が発生する
        """
        with pytest.raises(ValueError):
            ScorerWrapper(model_name="invalid-model")

    def test_predict_single_image(self, scorer, mock_image):
        """単一画像のスコア予測テスト

        期待される動作:
            - 画像が正しくスコアリングされる
            - スコアが適切な形式で返される

        Args:
            scorer: ScorerWrapperインスタンス（フィクスチャ）
            mock_image: モック化された画像（フィクスチャ）
        """
        with patch.object(scorer.scorer, "predict") as mock_predict:
            mock_predict.return_value = ["[WD]score_7"]
            result = scorer.predict(mock_image)
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0].startswith("[WD]score_")
            mock_predict.assert_called_once_with(mock_image)

    def test_predict_batch(self, scorer):
        """バッチ処理でのスコア予測テスト

        期待される動作:
            - 複数画像が一括でスコアリングされる
            - 各画像のスコアが適切な形式で返される

        Args:
            scorer: ScorerWrapperインスタンス（フィクスチャ）
        """
        mock_images = [MagicMock(spec=Image.Image) for _ in range(3)]
        expected_scores = [["[WD]score_7"], ["[WD]score_8"], ["[WD]score_6"]]

        with patch.object(scorer.scorer, "predict_pipe") as mock_predict_pipe:
            mock_predict_pipe.return_value = iter(expected_scores)
            results = list(scorer.predict_batch(mock_images))

            assert len(results) == len(mock_images)
            for result in results:
                assert isinstance(result, list)
                assert len(result) == 1
                assert result[0].startswith("[WD]score_")
            mock_predict_pipe.assert_called_once_with(mock_images)

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