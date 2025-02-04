from unittest.mock import patch, ANY

import pytest

from userscripts.taggers.cafeai_aesthetic_classifier import CafeAIAesthetic, BATCH_SIZE


class TestCafeAIAesthetic:
    """CafeAIAestheticタガーのテスト"""

    def test_init(self):
        """初期化のテスト - 名前の設定を確認"""
        tagger = CafeAIAesthetic()
        assert tagger.name() == "cafeai aesthetic classifier"

    def test_predict(self, test_image, mock_interrogator):
        """予測機能のテスト - スコアに基づくタグ生成を確認"""
        with patch(
            "userscripts.taggers.cafeai_aesthetic_classifier.pipeline",
            return_value=mock_interrogator,
        ):
            mock_interrogator.return_value = [
                {"label": "aesthetic", "score": 0.85},
                {"label": "not_aesthetic", "score": 0.15},
            ]

            tagger = CafeAIAesthetic()
            tagger.start()  # 必要なモデルをロード
            result = tagger.predict(test_image)
            assert result == ["[CAFE]score_8"]  # 0.85 * 10 = 8.5, floor = 8
            mock_interrogator.assert_called_once_with(test_image, top_k=2)

    def test_predict_different_scores(self, test_image, mock_interrogator):
        """予測機能のテスト - 異なるスコアでのタグ生成を確認"""
        with patch(
            "userscripts.taggers.cafeai_aesthetic_classifier.pipeline",
            return_value=mock_interrogator,
        ):
            test_cases = [
                (0.95, "[CAFE]score_9"),  # 9.5 -> 9
                (0.85, "[CAFE]score_8"),  # 8.5 -> 8
                (0.75, "[CAFE]score_7"),  # 7.5 -> 7
                (0.65, "[CAFE]score_6"),  # 6.5 -> 6
            ]

            tagger = CafeAIAesthetic()
            tagger.start()  # 必要なモデルをロード
            for score, expected_tag in test_cases:
                mock_interrogator.return_value = [
                    {"label": "aesthetic", "score": score},
                    {"label": "not_aesthetic", "score": 1 - score},
                ]
                result = tagger.predict(test_image)
                assert result == [expected_tag]

    def test_predict_pipe(self, test_image, mock_interrogator):
        """バッチ処理のテスト - 複数画像の処理を確認"""
        with patch(
            "userscripts.taggers.cafeai_aesthetic_classifier.pipeline",
            return_value=mock_interrogator,
        ):
            mock_interrogator.return_value = [
                [
                    {"label": "aesthetic", "score": 0.85},
                    {"label": "not_aesthetic", "score": 0.15},
                ]
            ]

            tagger = CafeAIAesthetic()
            tagger.start()  # 必要なモデルをロード
            results = list(tagger.predict_pipe([test_image]))
            assert len(results) == 1
            assert results[0] == ["[CAFE]score_8"]

    def test_predict_pipe_none_input(self):
        """バッチ処理のテスト - None入力の処理を確認"""
        tagger = CafeAIAesthetic()
        results = list(tagger.predict_pipe(None))
        assert len(results) == 0

    @pytest.fixture
    def mock_pipeline(self, mock_interrogator):
        """pipelineのモックを提供するfixture"""
        with patch(
            "userscripts.taggers.cafeai_aesthetic_classifier.pipeline",
            return_value=mock_interrogator,
        ) as mock_pipe:
            yield mock_pipe

    def test_start_stop(self, mock_pipeline, mock_interrogator):
        """start/stopメソッドのテスト"""
        tagger = CafeAIAesthetic()
        
        # start
        tagger.start()
        mock_pipeline.assert_called_once_with(
            "image-classification",
            "cafeai/cafe_aesthetic",
            device=ANY,
            batch_size=BATCH_SIZE
        )
        
        # stop
        with patch("userscripts.taggers.cafeai_aesthetic_classifier.settings") as mock_settings:
            mock_settings.current.interrogator_keep_in_memory = False
            tagger.stop()
            assert tagger.pipe_aesthetic is None