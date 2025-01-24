import re
from pathlib import Path

import pytest
from PIL import Image

from lib.scorer.wrapper import ScorerWrapper


@pytest.fixture
def test_image_dir():
    """テスト用画像ディレクトリを提供するフィクスチャ"""
    return Path("tests/resources/img/1_img")


@pytest.fixture
def sample_image(test_image_dir):
    """テスト用のサンプル画像を提供するフィクスチャ"""
    image_path = test_image_dir / "file01.webp"
    if not image_path.exists():
        pytest.skip("Sample image not found")
    return Image.open(image_path)


@pytest.fixture
def sample_images(test_image_dir):
    """バッチテスト用の複数サンプル画像を提供するフィクスチャ"""
    image_paths = [test_image_dir / "file01.webp", test_image_dir / "file02.webp"]
    if not all(path.exists() for path in image_paths):
        pytest.skip("Sample images not found")
    return [Image.open(path) for path in image_paths]


class TestScorerPipeline:
    """スコアリングパイプライン全体の統合テスト"""

    def _validate_score_format(self, score: str, prefix: str):
        """スコアのフォーマットを検証する

        Args:
            score: 検証するスコア文字列
            prefix: 期待されるプレフィックス（[WD], [CAFE], [IAP]など）
        """
        score_pattern = re.compile(f'\\{prefix}score_\\d+(\\.\\d+)?')
        assert score_pattern.match(score), f"Invalid score format: {score}"

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "model_name,score_prefix",
        [
            ("waifu-aesthetic", "[WD]"),
            ("aesthetic-shadow", "[AS]"),
            pytest.param("cafeai-aesthetic", "[CAFE]", marks=pytest.mark.slow),
            pytest.param("improved-aesthetic", "[IAP]", marks=pytest.mark.slow),
        ],
    )
    def test_single_image_prediction(self, model_name: str, score_prefix: str, sample_image):
        """単一画像のスコア予測パイプラインのテスト

        Args:
            model_name: テスト対象のモデル名
            score_prefix: 期待されるスコアのプレフィックス
            sample_image: テスト用画像
        """
        with ScorerWrapper(model_name) as scorer:
            # 予測実行
            scores = scorer.predict(sample_image)

            # 基本的な検証
            assert isinstance(scores, list)
            assert len(scores) > 0

            # スコアのフォーマット検証
            for score in scores:
                assert isinstance(score, str)
                self._validate_score_format(score, score_prefix)

    @pytest.mark.integration
    def test_batch_prediction(self, sample_images):
        """バッチ予測パイプラインのテスト

        Args:
            sample_images: テスト用画像のリスト
        """
        # バッチ処理対応モデルを使用
        model_name = "waifu-aesthetic"
        with ScorerWrapper(model_name) as scorer:
            try:
                # バッチ予測実行
                predictions = list(scorer.predict_batch(sample_images))

                # 基本的な検証
                assert len(predictions) == len(sample_images)

                # スコアのフォーマット検証
                for scores in predictions:
                    assert isinstance(scores, list)
                    assert len(scores) > 0
                    for score in scores:
                        assert isinstance(score, str)
                        self._validate_score_format(score, "[WD]")
            except (AttributeError, NotImplementedError):
                pytest.skip("バッチ処理はサポートされていません")

    @pytest.mark.integration
    def test_score_consistency(self, sample_image):
        """同一画像に対するスコアの一貫性テスト

        Args:
            sample_image: テスト用画像
        """
        model_name = "waifu-aesthetic"
        with ScorerWrapper(model_name) as scorer:
            # 同一画像に対して複数回予測
            scores1 = scorer.predict(sample_image)
            scores2 = scorer.predict(sample_image)

            # スコアの一貫性を検証
            assert scores1 == scores2, "Scores should be consistent for the same image"

    @pytest.mark.integration
    def test_resource_management(self, sample_image):
        """リソース管理の動作確認テスト

        Args:
            sample_image: テスト用画像
        """
        model_name = "waifu-aesthetic"
        scorer = ScorerWrapper(model_name)

        try:
            # コンテキストマネージャ外での使用
            scores1 = scorer.predict(sample_image)
            assert isinstance(scores1, list)

            # 明示的なリソース解放
            scorer.scorer.stop()

            # コンテキストマネージャでの使用
            with ScorerWrapper(model_name) as managed_scorer:
                scores2 = managed_scorer.predict(sample_image)
                assert isinstance(scores2, list)

            # コンテキスト終了後の自動リソース解放を確認
            assert not hasattr(managed_scorer.scorer, 'model') or managed_scorer.scorer.model is None

        finally:
            # クリーンアップ
            if hasattr(scorer, 'scorer'):
                scorer.scorer.stop()

    @pytest.mark.integration
    def test_invalid_model_handling(self):
        """無効なモデル名のハンドリングテスト"""
        invalid_model_name = "non-existent-model"
        with pytest.raises(ValueError, match=f"Unknown model name: {invalid_model_name}"):
            ScorerWrapper(invalid_model_name)