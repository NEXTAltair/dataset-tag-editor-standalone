"""ImprovedAestheticPredictorの単体テスト"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from transformers.testing_utils import TestCasePlus

from userscripts.taggers.improved_aesthetic_predictor import (
    Classifier,
    ImprovedAestheticPredictor,
)


def verify_model_device(model, expected_device):
    """モデルの全レイヤーのデバイスを確認"""
    for name, param in model.named_parameters():
        assert str(param.device) == expected_device, (
            f"Parameter {name} is on wrong device"
        )
    for name, buffer in model.named_buffers():
        assert str(buffer.device) == expected_device, (
            f"Buffer {name} is on wrong device"
        )


@pytest.fixture
def mock_iap_dependencies(mock_state_dict):
    """ImprovedAestheticPredictor の依存関係をモックする fixture"""
    with (
        patch(
            "userscripts.taggers.improved_aesthetic_predictor.CLIPProcessor"
        ) as mock_processor,
        patch(
            "userscripts.taggers.improved_aesthetic_predictor.CLIPModel"
        ) as mock_model,
        patch(
            "userscripts.taggers.improved_aesthetic_predictor.model_loader"
        ) as mock_loader,
        patch(
            "userscripts.taggers.improved_aesthetic_predictor.torch.load"
        ) as mock_torch_load,
        patch(
            "userscripts.taggers.improved_aesthetic_predictor.devices"
        ) as mock_devices,
    ):
        # モックの設定
        mock_processor.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_loader.load.return_value = "dummy_path"
        mock_torch_load.return_value = mock_state_dict
        mock_devices.device = torch.device("cpu")
        yield mock_processor, mock_model, mock_model_instance, mock_loader, mock_torch_load, mock_devices


class TestImprovedAestheticPredictor:
    """ImprovedAestheticPredictorタガーのテスト"""

    def test_init(self):
        """初期化のテスト - 名前の設定を確認"""
        tagger = ImprovedAestheticPredictor()
        assert tagger.name() == "Improved Aesthetic Predictor"

    @pytest.fixture
    def mock_state_dict(self):
        """モデルのstate_dictモックを提供するfixture"""
        return {
            "layers.0.weight": torch.randn(1024, 768),
            "layers.0.bias": torch.randn(1024),
            "layers.2.weight": torch.randn(128, 1024),
            "layers.2.bias": torch.randn(128),
            "layers.4.weight": torch.randn(64, 128),
            "layers.4.bias": torch.randn(64),
            "layers.6.weight": torch.randn(16, 64),
            "layers.6.bias": torch.randn(16),
            "layers.7.weight": torch.randn(1, 16),
            "layers.7.bias": torch.randn(1),
        }

    def test_predict(self, test_image, mock_state_dict, mock_iap_dependencies):
        """予測機能のテスト - スコアに基づくタグ生成を確認"""
        mock_processor, mock_model, mock_model_instance, mock_loader, mock_torch_load, mock_devices = mock_iap_dependencies

        # CLIPモデルの出力を設定
        features = torch.ones(1, 768)  # バッチ次元を含む
        features = features / torch.norm(features, dim=1, keepdim=True)  # 正規化
        mock_model_instance.get_image_features.return_value = features

        tagger = ImprovedAestheticPredictor()
        tagger.start()

        # モデルの出力を設定
        mock_classifier = MagicMock()
        mock_classifier.forward = MagicMock(return_value=torch.tensor([[8.5]]))
        mock_classifier.to.return_value = mock_classifier
        mock_classifier.named_parameters = MagicMock(return_value=[])
        mock_classifier.named_buffers = MagicMock(return_value=[])
        tagger.model = mock_classifier

        # デバイスの確認
        verify_model_device(tagger.model, "cpu")

        # スコアが8.5の場合に、タグが [IAP]score_8 となることを確認
        result = tagger.predict(test_image)
        assert result == ["[IAP]score_8"]

        # 予測後もデバイスが変わっていないことを確認
        verify_model_device(tagger.model, "cpu")

    def test_predict_different_scores(self, test_image, mock_state_dict, mock_iap_dependencies):
        """予測機能のテスト - 異なるスコアでのタグ生成を確認"""
        mock_processor, mock_model, mock_model_instance, mock_loader, mock_torch_load, mock_devices = mock_iap_dependencies

        test_cases = [
            (9.5, "[IAP]score_9"),  # 9.5 -> 9
            (8.5, "[IAP]score_8"),  # 8.5 -> 8
            (7.5, "[IAP]score_7"),  # 7.5 -> 7
            (6.5, "[IAP]score_6"),  # 6.5 -> 6
        ]

        # CLIPモデルの出力を設定
        features = torch.ones(1, 768)  # バッチ次元を含む
        features = features / torch.norm(features, dim=1, keepdim=True)  # 正規化
        mock_model_instance.get_image_features.return_value = features

        tagger = ImprovedAestheticPredictor()
        tagger.start()

        # モデルの出力を設定
        mock_classifier = MagicMock()
        mock_classifier.to.return_value = mock_classifier
        mock_classifier.named_parameters = MagicMock(return_value=[])
        mock_classifier.named_buffers = MagicMock(return_value=[])
        tagger.model = mock_classifier

        for score, expected_tag in test_cases:
            # デバイスの確認
            verify_model_device(tagger.model, "cpu")

            # スコアが異なる場合に、期待されるタグが生成されることを確認
            mock_classifier.forward = MagicMock(
                return_value=torch.tensor([[score]])
            )
            result = tagger.predict(test_image)
            assert result == [expected_tag]

            # 予測後もデバイスが変わっていないことを確認
            verify_model_device(tagger.model, "cpu")

    def test_predict_pipe(self, test_image, mock_state_dict, mock_iap_dependencies):
        """バッチ処理のテスト - 複数画像の処理を確認"""
        mock_processor, mock_model, mock_model_instance, mock_loader, mock_torch_load, mock_devices = mock_iap_dependencies

        # CLIPモデルの出力を設定（バッチ処理用）
        features = torch.ones(2, 768)  # バッチサイズ2
        features = features / torch.norm(features, dim=1, keepdim=True)  # 正規化
        mock_model_instance.get_image_features.return_value = features

        tagger = ImprovedAestheticPredictor()
        tagger.start()

        # モデルの出力を設定
        mock_classifier = MagicMock()
        mock_classifier.forward = MagicMock(
            return_value=torch.tensor([[8.5], [8.5]])
        )
        mock_classifier.to.return_value = mock_classifier
        mock_classifier.named_parameters = MagicMock(return_value=[])
        mock_classifier.named_buffers = MagicMock(return_value=[])
        tagger.model = mock_classifier

        # デバイスの確認
        verify_model_device(tagger.model, "cpu")

        # 複数画像を入力した場合に、バッチ処理で予測が実行され、期待されるタグが生成されることを確認
        results = list(tagger.predict_pipe([test_image, test_image]))
        assert len(results) == 2
        assert all(result == ["[IAP]score_8"] for result in results)

        # 予測後もデバイスが変わっていないことを確認
        verify_model_device(tagger.model, "cpu")

    def test_predict_pipe_none_input(self):
        """バッチ処理のNone入力テスト"""
        tagger = ImprovedAestheticPredictor()
        results = list(tagger.predict_pipe(None))
        assert len(results) == 0

    def test_predict_pipe_empty_list(self):
        """バッチ処理の空リスト入力テスト"""
        tagger = ImprovedAestheticPredictor()
        results = list(tagger.predict_pipe([]))
        assert len(results) == 0

    def test_start_stop(self, mock_state_dict, mock_iap_dependencies):
        """start/stopメソッドのテスト"""
        mock_processor, mock_model, mock_model_instance, mock_loader, mock_torch_load, mock_devices = mock_iap_dependencies

        tagger = ImprovedAestheticPredictor()

        # start
        tagger.start()
        mock_loader.load.assert_called_once()
        mock_processor.from_pretrained.assert_called_once_with(
            "openai/clip-vit-large-patch14"
        )
        mock_model.from_pretrained.assert_called_once_with(
            "openai/clip-vit-large-patch14"
        )

        # stop
        with patch(
            "userscripts.taggers.improved_aesthetic_predictor.settings"
        ) as mock_settings:
            mock_settings.current.interrogator_keep_in_memory = False
            tagger.stop()
            # start メソッドでロードされたモデルが stop メソッドで unload されることを確認
            assert tagger.model is None
            assert tagger.clip_processor is None
            assert tagger.clip_model is None


class TestClassifier:
    """Classifierクラスのテスト"""

    def test_init(self):
        """初期化のテスト - レイヤー構造の確認"""
        input_size = 768
        model = Classifier(input_size)

        # レイヤーの数と構造を確認
        layers = list(model.layers)
        assert len(layers) == 8  # 期待されるレイヤー数

        # 入力サイズと出力サイズの確認
        assert layers[0].in_features == input_size
        assert layers[-1].out_features == 1

    def test_forward(self):
        """forward passのテスト - 出力の形状を確認"""
        input_size = 768
        batch_size = 2
        model = Classifier(input_size)

        # テスト用の入力データ
        x = torch.randn(batch_size, input_size)
        output = model(x)

        # 出力のシェイプを確認
        assert output.shape == (batch_size, 1)
