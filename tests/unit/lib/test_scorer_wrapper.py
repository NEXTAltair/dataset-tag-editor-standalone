from unittest.mock import MagicMock, patch

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
            - 生データと加工済みデータが適切な形式で返される
            - 処理成功フラグが含まれる

        Args:
            scorer: ScorerWrapperインスタンス（フィクスチャ）
            mock_image: モック化された画像（フィクスチャ）
        """
        raw_data = [{'label': 'hq', 'score': 0.92}]
        formatted_data = ['very aesthetic']

        with patch.object(scorer.scorer, "predict") as mock_predict, \
             patch.object(scorer.scorer, "_get_score") as mock_get_score, \
             patch.object(scorer.scorer, "name") as mock_name:

            mock_predict.return_value = raw_data
            mock_get_score.return_value = formatted_data
            mock_name.return_value = 'test_model'

            result = scorer.predict(mock_image)

            assert result == {
                'raw_output': raw_data,
                'formatted_tags': formatted_data,
                'success': True,
                'model': 'test_model'
            }

            mock_predict.assert_called_once_with(mock_image)
            mock_get_score.assert_called_once_with(raw_data)

    def test_predict_batch(self, scorer):
        """バッチ処理でのスコア予測テスト

        期待される動作:
            - 複数画像が一括でスコアリングされる
            - 各画像のスコアが適切な形式で返される
            - バッチ処理の成功フラグが含まれる
            - 各予測結果が適切な形式である

        Args:
            scorer: ScorerWrapperインスタンス（フィクスチャ）
        """
        test_images = [
            Image.new('RGB', (256, 256)),
            Image.new('RGB', (256, 256)),
            Image.new('RGB', (256, 256))
        ]

        raw_data = [{'label': 'hq', 'score': 0.92}]
        formatted_data = ['very aesthetic']

        with patch.object(scorer.scorer, "predict") as mock_predict, \
             patch.object(scorer.scorer, "_get_score") as mock_get_score, \
             patch.object(scorer.scorer, "name") as mock_name:

            mock_predict.return_value = raw_data
            mock_get_score.return_value = formatted_data
            mock_name.return_value = 'test_model'

            result = scorer.predict_batch(test_images)

            # 基本的な構造の検証
            assert isinstance(result, dict)
            assert "results" in result
            assert "batch_success" in result
            assert result["batch_success"] is True
            assert len(result["results"]) == len(test_images)

            # 各予測結果の検証
            for prediction in result["results"]:
                assert isinstance(prediction, dict)
                assert "raw_output" in prediction
                assert "formatted_tags" in prediction
                assert "success" in prediction
                assert "model" in prediction
                assert prediction["success"] is True
                assert prediction["model"] == "test_model"
                assert prediction["raw_output"] == raw_data
                assert prediction["formatted_tags"] == formatted_data

            # モックの呼び出し回数を検証
            assert mock_predict.call_count == len(test_images)
            assert mock_get_score.call_count == len(test_images)

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