from unittest.mock import patch, MagicMock, ANY

import numpy as np
import pytest
import torch

from userscripts.taggers.waifu_aesthetic_classifier import (
    WaifuAesthetic,
    Classifier,
)


class TestWaifuAesthetic:
    """WaifuAestheticタガーのテスト"""

    def test_init(self):
        """初期化のテスト - 名前の設定を確認"""
        tagger = WaifuAesthetic()
        assert tagger.name() == "wd aesthetic classifier"

    def test_predict(self, test_image, mock_interrogator):
        """予測機能のテスト - スコアに基づくタグ生成を確認"""
        with patch("userscripts.taggers.waifu_aesthetic_classifier.CLIPProcessor") as mock_processor, \
             patch("userscripts.taggers.waifu_aesthetic_classifier.CLIPModel") as mock_model, \
             patch("userscripts.taggers.waifu_aesthetic_classifier.model_loader") as mock_loader, \
             patch("userscripts.taggers.waifu_aesthetic_classifier.torch.load") as mock_torch_load:

            # モックの設定
            mock_processor.from_pretrained.return_value = MagicMock()
            mock_model.from_pretrained.return_value = MagicMock()
            mock_loader.load.return_value = "dummy_path"
            mock_torch_load.return_value = {}

            # 特徴量とスコアの設定
            feature = np.array([0.5] * 512)
            score = torch.tensor([[0.85]])  # 0.85 * 10 = 8.5 -> 8
            
            # CLIPモデルの出力設定
            mock_model.from_pretrained.return_value.get_image_features.return_value = \
                torch.from_numpy(feature.reshape(1, -1))

            tagger = WaifuAesthetic()
            tagger.start()

            # モデルの出力を設定
            tagger.model = MagicMock()
            tagger.model.return_value = score

            result = tagger.predict(test_image)
            assert result == ["[WAIFU]score_8"]

    def test_predict_different_scores(self, test_image):
        """予測機能のテスト - 異なるスコアでのタグ生成を確認"""
        with patch("userscripts.taggers.waifu_aesthetic_classifier.CLIPProcessor") as mock_processor, \
             patch("userscripts.taggers.waifu_aesthetic_classifier.CLIPModel") as mock_model, \
             patch("userscripts.taggers.waifu_aesthetic_classifier.model_loader") as mock_loader, \
             patch("userscripts.taggers.waifu_aesthetic_classifier.torch.load") as mock_torch_load:

            # モックの設定
            mock_processor.from_pretrained.return_value = MagicMock()
            mock_model.from_pretrained.return_value = MagicMock()
            mock_loader.load.return_value = "dummy_path"
            mock_torch_load.return_value = {}

            test_cases = [
                (0.95, "[WAIFU]score_9"),  # 0.95 * 10 = 9.5 -> 9
                (0.85, "[WAIFU]score_8"),  # 0.85 * 10 = 8.5 -> 8
                (0.75, "[WAIFU]score_7"),  # 0.75 * 10 = 7.5 -> 7
                (0.65, "[WAIFU]score_6"),  # 0.65 * 10 = 6.5 -> 6
            ]

            feature = np.array([0.5] * 512)
            mock_model.from_pretrained.return_value.get_image_features.return_value = \
                torch.from_numpy(feature.reshape(1, -1))

            tagger = WaifuAesthetic()
            tagger.start()

            for score, expected_tag in test_cases:
                tagger.model = MagicMock()
                tagger.model.return_value = torch.tensor([[score]])
                result = tagger.predict(test_image)
                assert result == [expected_tag]

    def test_predict_pipe(self, test_image):
        """バッチ処理のテスト - 複数画像の処理を確認"""
        with patch("userscripts.taggers.waifu_aesthetic_classifier.CLIPProcessor") as mock_processor, \
             patch("userscripts.taggers.waifu_aesthetic_classifier.CLIPModel") as mock_model, \
             patch("userscripts.taggers.waifu_aesthetic_classifier.model_loader") as mock_loader, \
             patch("userscripts.taggers.waifu_aesthetic_classifier.torch.load") as mock_torch_load:

            # モックの設定
            mock_processor.from_pretrained.return_value = MagicMock()
            mock_model.from_pretrained.return_value = MagicMock()
            mock_loader.load.return_value = "dummy_path"
            mock_torch_load.return_value = {}

            # バッチ処理用の特徴量とスコアの設定
            feature = np.array([0.5] * 512)
            mock_model.from_pretrained.return_value.get_image_features.return_value = \
                torch.from_numpy(feature.reshape(1, -1))

            tagger = WaifuAesthetic()
            tagger.start()
            tagger.model = MagicMock()
            tagger.model.return_value = torch.tensor([[0.85]])  # 0.85 * 10 = 8.5 -> 8

            results = list(tagger.predict_pipe([test_image, test_image]))
            assert len(results) == 2
            assert all(result == ["[WAIFU]score_8"] for result in results)

    def test_predict_pipe_none_input(self):
        """バッチ処理のテスト - None入力の処理を確認"""
        tagger = WaifuAesthetic()
        results = list(tagger.predict_pipe(None))
        assert len(results) == 0

    def test_start_stop(self):
        """start/stopメソッドのテスト"""
        with patch("userscripts.taggers.waifu_aesthetic_classifier.CLIPProcessor") as mock_processor, \
             patch("userscripts.taggers.waifu_aesthetic_classifier.CLIPModel") as mock_model, \
             patch("userscripts.taggers.waifu_aesthetic_classifier.model_loader") as mock_loader, \
             patch("userscripts.taggers.waifu_aesthetic_classifier.torch.load") as mock_torch_load, \
             patch("userscripts.taggers.waifu_aesthetic_classifier.devices") as mock_devices:

            # モックの設定
            mock_processor.from_pretrained.return_value = MagicMock()
            mock_model.from_pretrained.return_value = MagicMock()
            mock_loader.load.return_value = "dummy_path"
            mock_torch_load.return_value = {}
            mock_devices.device = "cuda"

            tagger = WaifuAesthetic()
            
            # start
            tagger.start()
            mock_loader.load.assert_called_once()
            mock_processor.from_pretrained.assert_called_once_with("openai/clip-vit-base-patch32")
            mock_model.from_pretrained.assert_called_once_with("openai/clip-vit-base-patch32")
            
            # stop
            with patch("userscripts.taggers.waifu_aesthetic_classifier.settings") as mock_settings:
                mock_settings.current.interrogator_keep_in_memory = False
                tagger.stop()
                assert tagger.model is None
                assert tagger.clip_processor is None
                assert tagger.clip_model is None


class TestClassifier:
    """Classifierクラスのテスト"""

    def test_init(self):
        """初期化のテスト - レイヤー構造の確認"""
        input_size = 512
        hidden_size = 256
        output_size = 1
        model = Classifier(input_size, hidden_size, output_size)
        
        # レイヤーの構造を確認
        assert model.fc1.in_features == input_size
        assert model.fc1.out_features == hidden_size
        assert model.fc2.in_features == hidden_size
        assert model.fc2.out_features == hidden_size // 2
        assert model.fc3.in_features == hidden_size // 2
        assert model.fc3.out_features == output_size

    def test_forward(self):
        """forward passのテスト - 出力の形状とシグモイド関数の確認"""
        input_size = 512
        hidden_size = 256
        output_size = 1
        batch_size = 2
        model = Classifier(input_size, hidden_size, output_size)
        
        # テスト用の入力データ
        x = torch.randn(batch_size, input_size)
        output = model(x)
        
        # 出力のシェイプと値の範囲を確認
        assert output.shape == (batch_size, output_size)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # シグモイド関数の出力範囲