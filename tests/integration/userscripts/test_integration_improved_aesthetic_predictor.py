from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from userscripts.taggers.improved_aesthetic_predictor import (
    Classifier,
    ImprovedAestheticPredictor,
)


@pytest.fixture
def test_images():
    """テスト用の画像を提供するfixture"""
    image_paths = [
        "tests/resources/img/1_img/file01.webp",
        "tests/resources/img/1_img/file02.webp",
        "tests/resources/img/1_img/file03.webp",
    ]
    return [Image.open(path) for path in image_paths]


def test_classifier_architecture():
    """Classifierモデルのアーキテクチャテスト"""
    model = Classifier(768)

    # モデルの構造を確認
    assert len(model.layers) == 8
    assert isinstance(model.layers[0], torch.nn.Linear)
    assert model.layers[0].in_features == 768
    assert model.layers[0].out_features == 1024


def test_predict_single_image(test_images):
    """単一画像の予測テスト"""
    tagger = ImprovedAestheticPredictor()
    tagger.start()

    try:
        # 通常の予測テスト
        result = tagger.predict(test_images[0])
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], str)
        assert result[0].startswith("[IAP]score_")
        score = int(result[0].split("_")[1])
        assert 1 <= score <= 10

        # Wrapper経由の呼び出しテスト
        with patch.object(tagger, '_is_wrapper_call', return_value=True):
            result = tagger.predict(test_images[0])
            assert isinstance(result, tuple)
            assert len(result) == 2
            # 特徴量とスコアのタプルを返す
            features, scores = result
            assert isinstance(features, np.ndarray)  # 特徴量はnumpy.ndarray
            assert isinstance(scores, list)
            assert len(scores) == 1
            assert scores[0].startswith("[IAP]score_")

    finally:
        tagger.stop()


def test_predict_pipe_batch_processing(test_images):
    """バッチ処理のテスト"""
    tagger = ImprovedAestheticPredictor()
    tagger.start()

    try:
        # 通常のバッチ処理テスト
        batch_sizes = [None, 1, 2, len(test_images)]
        for batch_size in batch_sizes:
            results = list(tagger.predict_pipe(test_images, batch_size=batch_size))
            assert len(results) == len(test_images)
            for result in results:
                assert isinstance(result, list)
                assert len(result) == 1
                assert isinstance(result[0], str)
                assert result[0].startswith("[IAP]score_")
                score = int(result[0].split("_")[1])
                assert 1 <= score <= 10

        # Wrapper経由の呼び出しテスト
        with patch.object(tagger, '_is_wrapper_call', return_value=True):
            results = list(tagger.predict_pipe(test_images))
            assert len(results) == len(test_images)
            for result in results:
                assert isinstance(result, tuple)
                assert len(result) == 2
                # 特徴量とスコアのタプルを返す
                features, scores = result
                assert isinstance(features, np.ndarray)  # 特徴量はnumpy.ndarray
                assert isinstance(scores, list)
                assert len(scores) == 1
                assert scores[0].startswith("[IAP]score_")

    finally:
        tagger.stop()


def test_error_handling():
    """エラー処理のテスト"""
    tagger = ImprovedAestheticPredictor()
    tagger.start()

    try:
        # 無効な入力のテスト
        results = list(tagger.predict_pipe(None))
        assert len(results) == 0

        results = list(tagger.predict_pipe([]))
        assert len(results) == 0

        # 無効なバッチサイズのテスト
        with pytest.raises(ValueError):
            list(tagger.predict_pipe([Image.new("RGB", (64, 64))], batch_size=0))

        # chunksメソッドのエッジケースをテスト
        assert list(tagger.predict_pipe(None, batch_size=1)) == []
        assert list(tagger.predict_pipe([], batch_size=1)) == []

    finally:
        tagger.stop()


def test_device_management(devices):
    """デバイス管理のテスト"""
    # CPUデバイスのテスト
    devices.device = torch.device("cpu")
    tagger = ImprovedAestheticPredictor()
    tagger.start()
    
    # モデルの最初のパラメータのデバイスを確認
    param = next(tagger.model.parameters())
    assert str(param.device) == "cpu"
    tagger.stop()

    # CUDAデバイスのテスト（利用可能な場合）
    if torch.cuda.is_available():
        devices.device = torch.device("cuda:0")
        tagger = ImprovedAestheticPredictor()
        tagger.start()
        param = next(tagger.model.parameters())
        assert str(param.device) == "cuda:0"
        tagger.stop()


def test_tagger_name():
    """Taggerの名前テスト"""
    tagger = ImprovedAestheticPredictor()
    assert tagger.name() == "Improved Aesthetic Predictor"
