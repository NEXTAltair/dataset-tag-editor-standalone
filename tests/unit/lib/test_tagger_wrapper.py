from unittest.mock import Mock, patch

import pytest
from PIL import Image

from lib.tagger.wrapper import TaggerWrapper


class TestTaggerWrapper:
    """TaggerWrapperクラスのテストスイート

    このテストスイートでは、scripts/dataset_tag_editor/taggers_builtin.pyの
    Taggerクラスをラップする機能をテストします。
    主に以下の機能をテストします：
    - 各種タガーの初期化と管理
    - 画像からのタグ予測
    - スレッショルドによるフィルタリング
    - バッチ処理の動作
    """

    @pytest.fixture
    def mock_image(self):
        """テスト用の画像モックを作成するフィクスチャ

        Returns:
            Mock: PIL.Image型のモックオブジェクト
        """
        return Mock(spec=Image.Image)

    @pytest.fixture
    def mock_tagger(self):
        """モックTaggerインスタンスを作成するフィクスチャ

        Returns:
            Mock: Taggerクラスのモックオブジェクト
        """
        tagger = Mock()
        tagger.predict.return_value = ["tag1", "tag2"]
        tagger.name.return_value = "MockTagger"
        return tagger

    def test_init_with_valid_model(self):
        """有効なモデル名での初期化テスト

        期待される動作:
            - 指定されたモデル名でTaggerWrapperが正常に初期化される
            - start()メソッドが呼び出される
        """
        with patch("lib.tagger.wrapper.WaifuDiffusion") as mock_tagger_class:
            mock_tagger = Mock()
            mock_tagger_class.return_value = mock_tagger

            wrapper = TaggerWrapper("waifu-diffusion", threshold=0.5)

            mock_tagger_class.assert_called_once_with("waifu-diffusion", threshold=0.5)
            mock_tagger.start.assert_called_once()

    def test_init_with_invalid_model(self):
        """無効なモデル名での初期化テスト

        期待される動作:
            - 存在しないモデル名を指定した場合、ValueError例外が発生する
        """
        with pytest.raises(ValueError):
            TaggerWrapper("non-existent-model")

    def test_predict_single_image(self, mock_image, mock_tagger):
        """単一画像のタグ予測テスト

        期待される動作:
            - 画像に対して適切なタグが生成される
            - タグは文字列のリストとして返される

        Args:
            mock_image: モック化された画像オブジェクト
            mock_tagger: モック化されたTaggerインスタンス
        """
        with patch("lib.tagger.wrapper.WaifuDiffusion") as mock_tagger_class:
            mock_tagger_class.return_value = mock_tagger
            wrapper = TaggerWrapper("waifu-diffusion", threshold=0.5)

            tags = wrapper.predict(mock_image)

            assert isinstance(tags, list)
            assert all(isinstance(tag, str) for tag in tags)
            mock_tagger.predict.assert_called_once_with(mock_image, threshold=0.5)

    def test_predict_with_custom_threshold(self, mock_image, mock_tagger):
        """カスタムスレッショルドでのタグ予測テスト

        期待される動作:
            - 指定されたスレッショルドでタグがフィルタリングされる
            - デフォルトのスレッショルドが上書きされる

        Args:
            mock_image: モック化された画像オブジェクト
            mock_tagger: モック化されたTaggerインスタンス
        """
        with patch("lib.tagger.wrapper.WaifuDiffusion") as mock_tagger_class:
            mock_tagger_class.return_value = mock_tagger
            wrapper = TaggerWrapper("waifu-diffusion", threshold=0.5)

            custom_threshold = 0.7
            wrapper.predict(mock_image, threshold=custom_threshold)

            mock_tagger.predict.assert_called_once_with(
                mock_image, threshold=custom_threshold
            )

    def test_predict_batch(self, mock_image, mock_tagger):
        """バッチ処理でのタグ予測テスト

        期待される動作:
            - 複数画像に対して一括でタグ予測が行われる
            - 各画像に対するタグのリストが返される

        Args:
            mock_image: モック化された画像オブジェクト
            mock_tagger: モック化されたTaggerインスタンス
        """
        with patch("lib.tagger.wrapper.WaifuDiffusionTimm") as mock_tagger_class:
            mock_tagger.predict_pipe.return_value = iter(
                [["tag1", "tag2"], ["tag3", "tag4"]]
            )
            mock_tagger_class.return_value = mock_tagger

            wrapper = TaggerWrapper("waifu-diffusion-timm", threshold=0.5, batch_size=2)
            images = [mock_image] * 2

            tags = list(wrapper.predict_batch(images))

            assert len(tags) == 2
            assert all(isinstance(tag_list, list) for tag_list in tags)
            assert all(isinstance(tag, str) for tag_list in tags for tag in tag_list)
            mock_tagger.predict_pipe.assert_called_once()

    def test_context_manager(self, mock_tagger):
        """コンテキストマネージャとしての動作テスト

        期待される動作:
            - with文でのリソース管理が適切に行われる
            - 終了時にstop()メソッドが呼び出される

        Args:
            mock_tagger: モック化されたTaggerインスタンス
        """
        with patch("lib.tagger.wrapper.WaifuDiffusion") as mock_tagger_class:
            mock_tagger_class.return_value = mock_tagger

            with TaggerWrapper("waifu-diffusion", threshold=0.5) as wrapper:
                pass

            mock_tagger.start.assert_called_once()
            mock_tagger.stop.assert_called_once()

    def test_get_model_name(self, mock_tagger):
        """モデル名取得テスト

        期待される動作:
            - name()メソッドが正しく呼び出される
            - UIに表示する名前が取得できる

        Args:
            mock_tagger: モック化されたTaggerインスタンス
        """
        with patch("lib.tagger.wrapper.WaifuDiffusion") as mock_tagger_class:
            mock_tagger_class.return_value = mock_tagger
            wrapper = TaggerWrapper("waifu-diffusion", threshold=0.5)

            name = wrapper.get_model_name()

            assert isinstance(name, str)
            assert name == "MockTagger"
            mock_tagger.name.assert_called_once()

    def test_get_available_models(self):
        """使用可能なモデル一覧取得テスト

        期待される動作:
            - 使用可能なモデル名とその説明が辞書形式で取得できる
            - 全ての主要なモデルタイプが含まれている
            - カスタムモデルも含まれている
        """
        models = TaggerWrapper.get_available_models()
        
        # 基本的な型チェック
        assert isinstance(models, dict)
        assert all(isinstance(desc, str) for desc in models.values())
        
        # ビルトインモデル - 画像タグ付け
        assert "waifu-diffusion-*" in models
        assert "waifu-diffusion-*-timm" in models
        assert "deep-danbooru" in models
        assert "z3d-e621" in models
        
        # ビルトインモデル - 説明文生成
        assert "blip" in models
        assert "blip2-*" in models
        assert "git-large" in models
        
        # 説明文に具体的な情報が含まれていることを確認
        assert any("SmilingWolf" in desc for desc in models.values())
        assert any("Salesforce" in desc for desc in models.values())
        assert not any("美的評価" in desc for desc in models.values())  # 美的評価はScorerWrapperに移動

    def test_init_with_invalid_model_shows_available_models(self):
        """無効なモデル名での初期化時に利用可能なモデル一覧を表示するテスト

        期待される動作:
            - 存在しないモデル名を指定した場合、ValueError例外が発生する
            - エラーメッセージに利用可能なモデル一覧が含まれる
        """
        with pytest.raises(ValueError) as exc_info:
            TaggerWrapper("non-existent-model")
        
        error_message = str(exc_info.value)
        assert "Unknown model name: non-existent-model" in error_message
        assert "Available models:" in error_message
        assert "waifu-diffusion-*" in error_message
