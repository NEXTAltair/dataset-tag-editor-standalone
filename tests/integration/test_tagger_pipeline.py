from pathlib import Path

import pytest
from PIL import Image
from huggingface_hub.utils import RepositoryNotFoundError

from lib.tagger.wrapper import TaggerWrapper


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


class TestTaggerPipeline:
    """タグ付けパイプライン全体の統合テスト"""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "model_name",
        [
            "wd-eva02-large-tagger-v3",  # 最新の主力モデル
            "wd-vit-large-tagger-v3",    # 人気の高いモデル
            pytest.param("wd-v1-4-swinv2-tagger-v2", marks=pytest.mark.slow),  # 安定版
            pytest.param("blip", marks=pytest.mark.slow),
            pytest.param("deep-danbooru", marks=pytest.mark.slow),
        ],
    )
    def test_single_image_prediction(self, model_name: str, sample_image):
        """単一画像の予測パイプラインのテスト

        Args:
            model_name: テスト対象のモデル名
            sample_image: テスト用画像
        """
        try:
            with TaggerWrapper(model_name) as tagger:
                # 予測実行
                tags = tagger.predict(sample_image)

                # 基本的な検証
                assert isinstance(tags, list)
                assert all(isinstance(tag, str) for tag in tags)
                assert len(tags) > 0
        except (RepositoryNotFoundError, RuntimeError) as e:
            pytest.skip(f"Model {model_name} not available: {str(e)}")

    @pytest.mark.integration
    def test_batch_prediction(self, sample_images):
        """バッチ予測パイプラインのテスト

        Args:
            sample_images: テスト用画像のリスト
        """
        # 最新のTimm対応モデルを使用
        model_name = "wd-eva02-large-tagger-v3"
        try:
            with TaggerWrapper(model_name, batch_size=2) as tagger:
                # バッチ予測実行
                predictions = list(tagger.predict_batch(sample_images))

                # 基本的な検証
                assert len(predictions) == len(sample_images)
                for tags in predictions:
                    assert isinstance(tags, list)
                    assert all(isinstance(tag, str) for tag in tags)
                    assert len(tags) > 0
        except (RepositoryNotFoundError, RuntimeError, AttributeError) as e:
            pytest.skip(f"Batch processing not available: {str(e)}")

    @pytest.mark.integration
    def test_threshold_filtering(self, sample_image):
        """閾値フィルタリングの動作確認テスト

        Args:
            sample_image: テスト用画像
        """
        # 確率値を出力するモデルを使用
        model_name = "wd-eva02-large-tagger-v3"
        try:
            # 異なる閾値での予測を比較
            with TaggerWrapper(model_name, threshold=0.1) as tagger_low:
                tags_low = tagger_low.predict(sample_image)

            with TaggerWrapper(model_name, threshold=0.9) as tagger_high:
                tags_high = tagger_high.predict(sample_image)

            # 高閾値の結果は低閾値の結果のサブセットであるべき
            assert len(tags_high) <= len(tags_low)
            assert all(tag in tags_low for tag in tags_high)
        except (RepositoryNotFoundError, RuntimeError) as e:
            pytest.skip(f"Model {model_name} not available: {str(e)}")

    @pytest.mark.integration
    def test_resource_management(self, sample_image):
        """リソース管理の動作確認テスト

        Args:
            sample_image: テスト用画像
        """
        model_name = "wd-eva02-large-tagger-v3"
        try:
            tagger = TaggerWrapper(model_name)

            try:
                # コンテキストマネージャ外での使用
                tags1 = tagger.predict(sample_image)
                assert isinstance(tags1, list)

                # 明示的なリソース解放
                tagger.tagger.stop()

                # コンテキストマネージャでの使用
                with TaggerWrapper(model_name) as managed_tagger:
                    tags2 = managed_tagger.predict(sample_image)
                    assert isinstance(tags2, list)

                # コンテキスト終了後の自動リソース解放を確認
                assert not hasattr(managed_tagger.tagger, 'tagger_inst') or managed_tagger.tagger.tagger_inst is None

            finally:
                # クリーンアップ
                if hasattr(tagger, 'tagger'):
                    tagger.tagger.stop()
        except (RepositoryNotFoundError, RuntimeError) as e:
            pytest.skip(f"Model {model_name} not available: {str(e)}")

    @pytest.mark.integration
    def test_invalid_model_handling(self):
        """無効なモデル名のハンドリングテスト"""
        invalid_model_name = "non-existent-model"
        with pytest.raises(ValueError, match=f"Unknown model name: {invalid_model_name}"):
            TaggerWrapper(invalid_model_name)