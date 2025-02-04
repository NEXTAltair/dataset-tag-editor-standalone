from unittest.mock import patch, ANY

import pytest

from userscripts.taggers.aesthetic_shadow_v1 import AestheticShadow


class TestAestheticShadowV1:
    """AestheticShadowV1タガーのテスト"""

    def test_init(self):
        """初期化のテスト - 名前の設定を確認"""
        tagger = AestheticShadow()
        assert tagger.name() == "aesthetic shadow v1"

    def test_predict(self, test_image, mock_interrogator):
        """予測機能のテスト - スコアに基づくタグ生成を確認"""
        with patch(
            "userscripts.taggers.aesthetic_shadow_v1.pipeline",
            return_value=mock_interrogator,
        ):
            mock_interrogator.return_value = [
                {"label": "hq", "score": 0.8},  # very aesthetic
                {"label": "lq", "score": 0.2},
            ]

            tagger = AestheticShadow()
            tagger.start()  # 必要なモデルをロード
            result = tagger.predict(test_image)
            assert result == ["v1_very aesthetic"]
            mock_interrogator.assert_called_once_with(test_image)

    def test_predict_different_scores(self, test_image, mock_interrogator):
        """予測機能のテスト - 異なるスコアでのタグ生成を確認"""
        with patch(
            "userscripts.taggers.aesthetic_shadow_v1.pipeline",
            return_value=mock_interrogator,
        ):
            test_cases = [
                (0.9, "v1_very aesthetic"),   # > 0.71
                (0.6, "v1_aesthetic"),        # > 0.45
                (0.3, "v1_displeasing"),      # > 0.27
                (0.2, "v1_very displeasing"), # <= 0.27
            ]

            tagger = AestheticShadow()
            tagger.start()  # 必要なモデルをロード
            for score, expected_tag in test_cases:
                mock_interrogator.return_value = [
                    {"label": "hq", "score": score},
                    {"label": "lq", "score": 1 - score},
                ]
                result = tagger.predict(test_image)
                assert result == [expected_tag]

    def test_predict_pipe(self, test_image, mock_interrogator):
        """バッチ処理のテスト - 複数画像の処理を確認"""
        with patch(
            "userscripts.taggers.aesthetic_shadow_v1.pipeline",
            return_value=mock_interrogator,
        ):
            mock_interrogator.return_value = [
                [
                    {"label": "hq", "score": 0.8},  # very aesthetic
                    {"label": "lq", "score": 0.2},
                ]
            ]

            tagger = AestheticShadow()
            tagger.start()  # 必要なモデルをロード
            results = list(tagger.predict_pipe([test_image]))
            assert len(results) == 1
            assert results[0] == ["v1_very aesthetic"]

    def test_predict_pipe_none_input(self):
        """バッチ処理のテスト - None入力の処理を確認"""
        tagger = AestheticShadow()
        results = list(tagger.predict_pipe(None))
        assert len(results) == 0

    @pytest.fixture
    def mock_pipeline(self, mock_interrogator):
        """pipelineのモックを提供するfixture"""
        with patch(
            "userscripts.taggers.aesthetic_shadow_v1.pipeline",
            return_value=mock_interrogator,
        ) as mock_pipe:
            yield mock_pipe

    def test_start_stop(self, mock_pipeline, mock_interrogator):
        """start/stopメソッドのテスト"""
        tagger = AestheticShadow()
        
        # start
        tagger.start()
        mock_pipeline.assert_called_once_with(
            "image-classification",
            "shadowlilac/aesthetic-shadow",
            device=ANY,  # unittest.mock.ANYを使用
            batch_size=1
        )
        
        # stop
        with patch("userscripts.taggers.aesthetic_shadow_v1.settings") as mock_settings:
            mock_settings.current.interrogator_keep_in_memory = False
            tagger.stop()
            assert tagger.pipe_aesthetic is None