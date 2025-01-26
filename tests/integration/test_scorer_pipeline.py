from pathlib import Path
import pytest
from PIL import Image

from lib.scorer.wrapper import ScorerWrapper, ScorerPrediction, BatchScorerOutput


@pytest.fixture
def test_image_dir():
    """テスト用画像ディレクトリを提供するフィクスチャ
    
    Returns:
        Path: テスト用画像が格納されているディレクトリのパス
    """
    return Path("tests/resources/img/1_img")


@pytest.fixture
def sample_image(test_image_dir):
    """テスト用のサンプル画像を提供するフィクスチャ
    
    Args:
        test_image_dir (Path): テスト用画像ディレクトリのパス
        
    Returns:
        Image: 読み込まれたPIL Imageオブジェクト
        
    Raises:
        pytest.skip: サンプル画像が見つからない場合
    """
    image_path = test_image_dir / "file01.webp"
    if not image_path.exists():
        pytest.skip("Sample image not found")
    return Image.open(image_path)


@pytest.fixture
def sample_images(test_image_dir):
    """バッチテスト用の複数サンプル画像を提供するフィクスチャ
    
    Args:
        test_image_dir (Path): テスト用画像ディレクトリのパス
        
    Returns:
        list[Image]: 読み込まれたPIL Imageオブジェクトのリスト
        
    Raises:
        pytest.skip: サンプル画像が見つからない場合
    """
    image_paths = [test_image_dir / f"file0{i}.webp" for i in range(1, 4)]
    if not all(path.exists() for path in image_paths):
        pytest.skip("Sample images not found")
    return [Image.open(path) for path in image_paths]


class TestScorerPipeline:
    """スコアリングパイプライン全体の統合テスト"""

    def _validate_prediction(self, prediction: ScorerPrediction):
        """予測結果を検証する
        
        Args:
            prediction (ScorerPrediction): 検証する予測結果
            
        Raises:
            AssertionError: 予測結果の形式が期待される形式と異なる場合
        """
        # 基本的な構造検証
        assert isinstance(prediction, dict), "予測結果は辞書形式である必要があります"
        required_keys = {'raw_output', 'formatted_tags', 'success'}
        assert required_keys.issubset(prediction.keys()), "必要なキーが不足しています"
        
        # 型チェック
        assert isinstance(prediction['success'], bool), "successはbool型である必要があります"
        if prediction['success']:
            assert isinstance(prediction['formatted_tags'], list), "formatted_tagsはリストである必要があります"
            assert all(isinstance(tag, str) for tag in prediction['formatted_tags']), "タグは文字列である必要があります"
            assert len(prediction['formatted_tags']) > 0, "成功時はタグが存在する必要があります"
        else:
            assert 'error' in prediction, "失敗時はerrorキーが必要です"
            assert isinstance(prediction['error'], str), "エラーメッセージは文字列である必要があります"

    @pytest.mark.parametrize("model_name", [
        "waifu-aesthetic",
        "aesthetic-shadow",
        pytest.param("cafeai-aesthetic", marks=pytest.mark.slow),
        pytest.param("improved-aesthetic", marks=pytest.mark.slow)
    ])
    def test_scorer_pipeline(self, model_name: str, sample_image, sample_images):
        """スコアリングパイプラインの総合テスト
        
        以下の機能を検証します：
        1. 単一画像の予測
        2. バッチ予測
        3. リソース管理
        4. 予測結果のフォーマット検証
        """

        # 単一画像テスト
        with ScorerWrapper(model_name) as scorer:
            # 予測実行
            prediction = scorer.predict(sample_image)
            self._validate_prediction(prediction)
            
            # モデル名取得テスト
            model_name_display = scorer.get_model_name()
            assert isinstance(model_name_display, str), "モデル名は文字列である必要があります"
            assert len(model_name_display) > 0, "モデル名が空です"

        # バッチ予測テスト
        with ScorerWrapper(model_name, batch_size=2) as scorer:
            try:
                batch_result = scorer.predict_batch(sample_images)
                
                # バッチ結果の基本検証
                assert isinstance(batch_result, dict), "バッチ結果は辞書形式である必要があります"
                assert 'results' in batch_result, "resultsキーが必要です"
                assert 'batch_success' in batch_result, "batch_successキーが必要です"
                assert isinstance(batch_result['results'], list), "resultsはリストである必要があります"
                assert len(batch_result['results']) == len(sample_images), "結果の数が画像数と一致しません"
                
                # 個々の予測結果検証
                for result in batch_result['results']:
                    self._validate_prediction(result)

            except NotImplementedError:
                pytest.skip("このモデルはバッチ処理をサポートしていません")

        # リソース管理テスト
        scorer = ScorerWrapper(model_name)
        try:
            # 明示的なリソース管理テスト
            prediction = scorer.predict(sample_image)
            self._validate_prediction(prediction)
            scorer.scorer.stop()
            
            # コンテキストマネージャテスト
            with ScorerWrapper(model_name) as managed_scorer:
                managed_prediction = managed_scorer.predict(sample_image)
                self._validate_prediction(managed_prediction)
            
            # リソース解放確認
            assert not hasattr(managed_scorer.scorer, 'model') or managed_scorer.scorer.model is None, "リソースが適切に解放されていません"

        finally:
            if hasattr(scorer, 'scorer'):
                scorer.scorer.stop()

    def test_invalid_model_handling(self):
        """無効なモデル名のハンドリングテスト"""
        invalid_model = "non-existent-model"
        with pytest.raises(ValueError) as exc_info:
            ScorerWrapper(invalid_model)
            
        error_msg = str(exc_info.value)
        assert invalid_model in error_msg, "エラーメッセージにモデル名が含まれていません"
        assert "Available models" in error_msg, "利用可能モデル一覧が表示されていません"