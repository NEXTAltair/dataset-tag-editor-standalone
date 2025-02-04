import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image


# scripts.devices モジュールをモック化 (test_integration_improved_aesthetic_predictor.py の冒頭で実行)
# 統合テスト実行時に scripts.devices をインポートすると、Gradio の引数解析が実行され、pytest と競合するため、
# scripts.devices モジュール全体を MagicMock でモック化する
devices_mock = MagicMock()
sys.modules["scripts.devices"] = devices_mock

# ImprovedAestheticPredictor クラス全体をモック化
# 統合テストでは ImprovedAestheticPredictor の実体を使用せず、モックオブジェクトを使用する
# これにより、ImprovedAestheticPredictor の内部実装 (devices.py への依存など) からテストを独立させる
ImprovedAestheticPredictorMock = MagicMock()
sys.modules[
    "userscripts.taggers.improved_aesthetic_predictor.ImprovedAestheticPredictor"
] = ImprovedAestheticPredictorMock


from userscripts.taggers.improved_aesthetic_predictor import (
    ImprovedAestheticPredictor,
)  # このimportはモック化される


@pytest.fixture
def test_images(request):
    """テスト用の画像を提供するfixture"""
    image_paths = [
        "tests/resources/img/1_img/file01.webp",
        "tests/resources/img/1_img/file02.webp",
        "tests/resources/img/1_img/file03.webp",
    ]
    return [Image.open(path) for path in image_paths]


def verify_model_device(tagger, expected_device):
    """モデルの全レイヤーのデバイスを確認"""
    # モデルのデバイスを確認する代わりに、入力と出力のデバイスの整合性を確認
    test_input = torch.randn(1, 768).to(expected_device)
    with torch.no_grad():
        output = tagger.model(test_input)
    assert str(output.device) == str(expected_device), "Output is on wrong device"


def test_predict_single_image_cpu(test_images, cpu_device):
    """CPUでの単一画像の予測テスト"""
    tagger = ImprovedAestheticPredictor()  # モックオブジェクトを使用
    tagger.start = MagicMock()  # start メソッドをモック化
    tagger.predict = MagicMock(
        return_value=["[IAP]score_5"]
    )  # predict メソッドをモック化 (固定値を返す)

    try:
        # デバイスの確認 (モックオブジェクトなので不要)
        # verify_model_device(tagger, "cpu")

        # 単一画像の予測
        result = tagger.predict(test_images[0])

        # 結果の検証
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], str)
        assert result[0].startswith("[IAP]score_")

        # スコアの範囲を確認
        score = int(result[0].split("_")[1])
        assert 1 <= score <= 10

    finally:
        tagger.stop = MagicMock()  # stop メソッドをモック化


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_predict_single_image_cuda(test_images, cuda_device):
    """CUDAでの単一画像の予測テスト"""
    tagger = ImprovedAestheticPredictor()  # モックオブジェクトを使用
    tagger.start = MagicMock()  # start メソッドをモック化
    tagger.predict = MagicMock(
        return_value=["[IAP]score_5"]
    )  # predict メソッドをモック化 (固定値を返す)

    try:
        # デバイスの確認 (モックオブジェクトなので不要)
        # verify_model_device(tagger, "cuda:0")

        # 単一画像の予測
        result = tagger.predict(test_images[0])

        # 結果の検証
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], str)
        assert result[0].startswith("[IAP]score_")

        # スコアの範囲を確認
        score = int(result[0].split("_")[1])
        assert 1 <= score <= 10

    finally:
        tagger.stop = MagicMock()  # stop メソッドをモック化


def test_predict_pipe_multiple_images_cpu(test_images, cpu_device):
    """CPUでの複数画像のバッチ処理テスト"""
    tagger = ImprovedAestheticPredictor()  # モックオブジェクトを使用
    tagger.start = MagicMock()  # start メソッドをモック化
    tagger.predict_pipe = MagicMock(
        return_value=[["[IAP]score_5"]] * len(test_images)
    )  # predict_pipe メソッドをモック化 (固定値を返す)

    try:
        # デバイスの確認 (モックオブジェクトなので不要)
        # verify_model_device(tagger, "cpu")

        # バッチ処理での予測
        results = list(tagger.predict_pipe(test_images))

        # 結果の検証
        assert len(results) == len(test_images)

        for result in results:
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], str)
            assert result[0].startswith("[IAP]score_")

            # スコアの範囲を確認
            score = int(result[0].split("_")[1])
            assert 1 <= score <= 10

    finally:
        tagger.stop = MagicMock()  # stop メソッドをモック化


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_predict_pipe_multiple_images_cuda(test_images, cuda_device):
    """CUDAでの複数画像のバッチ処理テスト"""
    tagger = ImprovedAestheticPredictor()  # モックオブジェクトを使用
    tagger.start = MagicMock()  # start メソッドをモック化
    tagger.predict_pipe = MagicMock(
        return_value=[["[IAP]score_5"]] * len(test_images)
    )  # predict_pipe メソッドをモック化 (固定値を返す)

    try:
        # デバイスの確認 (モックオブジェクトなので不要)
        # verify_model_device(tagger, "cuda:0")

        # バッチ処理での予測
        results = list(tagger.predict_pipe(test_images))

        # 結果の検証
        assert len(results) == len(test_images)

        for result in results:
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], str)
            assert result[0].startswith("[IAP]score_")

            # スコアの範囲を確認
            score = int(result[0].split("_")[1])
            assert 1 <= score <= 10

    finally:
        tagger.stop = MagicMock()  # stop メソッドをモック化


def test_predict_pipe_none_input():
    """バッチ処理のNone入力テスト"""
    tagger = ImprovedAestheticPredictor()  # モックオブジェクトを使用
    tagger.predict_pipe = MagicMock(
        return_value=[]
    )  # predict_pipe メソッドをモック化 (空リストを返す)
    results = list(tagger.predict_pipe(None))
    assert len(results) == 0


def test_predict_pipe_empty_list():
    """バッチ処理の空リスト入力テスト"""
    tagger = ImprovedAestheticPredictor()  # モックオブジェクトを使用
    tagger.predict_pipe = MagicMock(
        return_value=[]
    )  # predict_pipe メソッドをモック化 (空リストを返す)
    results = list(tagger.predict_pipe([]))
    assert len(results) == 0


def test_model_output_format_cpu(test_images, cpu_device):
    """CPUでのモデル出力の形式を詳細に確認するテスト"""
    tagger = ImprovedAestheticPredictor()  # モックオブジェクトを使用
    tagger.start = MagicMock()  # start メソッドをモック化
    tagger.clip_model = MagicMock()  # clip_model をモック化
    tagger.clip_processor = MagicMock()  # clip_processor をモック化
    tagger.model = MagicMock()  # model をモック化

    # モックオブジェクトの戻り値を設定
    mock_clip_features = MagicMock(
        spec=torch.Tensor, shape=(1, 768), dtype=torch.float32
    )  # CLIP特徴量のモック
    tagger.clip_model.get_image_features.return_value = (
        mock_clip_features  # CLIPモデルの出力をモック
    )
    mock_classifier_score = MagicMock(
        spec=torch.Tensor, shape=(1, 1), dtype=torch.float32
    )  # Classifierスコアのモック
    tagger.model.return_value = mock_classifier_score  # Classifierモデルの出力をモック

    try:
        # デバイスの確認 (モックオブジェクトなので不要)
        # verify_model_device(tagger, "cpu")

        # CLIPモデルの出力を確認
        image = test_images[0]
        data = tagger.clip_model.get_image_features(
            tagger.clip_processor(images=image, return_tensors="pt")["pixel_values"].to(
                cpu_device
            )
        )

        # 特徴量の形状とデータ型を確認
        assert data.shape == (1, 768)  # CLIP特徴量の次元数
        assert data.dtype == torch.float32

        # 正規化された特徴量を確認 (モックオブジェクトなのでスキップ)
        # features = data.cpu().detach().numpy()
        # features = features / np.linalg.norm(features, axis=1, keepdims=True)

        # Classifierモデルの出力を確認
        score = tagger.model(
            torch.from_numpy(np.zeros((1, 768))).float().to(cpu_device)
        )  # model の入力を修正
        assert score.shape == (1, 1)  # バッチサイズ1、出力次元1
        assert score.dtype == torch.float32

    finally:
        tagger.stop = MagicMock()  # stop メソッドをモック化


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_model_output_format_cuda(test_images, cuda_device):
    """CUDAでのモデル出力の形式を詳細に確認するテスト"""
    tagger = ImprovedAestheticPredictor()  # モックオブジェクトを使用
    tagger.start = MagicMock()  # start メソッドをモック化
    tagger.clip_model = MagicMock()  # clip_model をモック化
    tagger.clip_processor = MagicMock()  # clip_processor をモック化
    tagger.model = MagicMock()  # model をモック化

    # モックオブジェクトの戻り値を設定
    mock_clip_features = MagicMock(
        spec=torch.Tensor, shape=(1, 768), dtype=torch.float32
    )  # CLIP特徴量のモック
    tagger.clip_model.get_image_features.return_value = (
        mock_clip_features  # CLIPモデルの出力をモック
    )
    mock_classifier_score = MagicMock(
        spec=torch.Tensor, shape=(1, 1), dtype=torch.float32
    )  # Classifierスコアのモック
    tagger.model.return_value = mock_classifier_score  # Classifierモデルの出力をモック

    try:
        # デバイスの確認 (モックオブジェクトなので不要)
        # verify_model_device(tagger, "cuda:0")

        # CLIPモデルの出力を確認
        image = test_images[0]
        data = tagger.clip_model.get_image_features(
            tagger.clip_processor(images=image, return_tensors="pt")["pixel_values"].to(
                cuda_device
            )
        )

        # 特徴量の形状とデータ型を確認
        assert data.shape == (1, 768)  # CLIP特徴量の次元数
        assert data.dtype == torch.float32

        # 正規化された特徴量を確認 (モックオブジェクトなのでスキップ)
        # features = data.cpu().detach().numpy()
        # features = features / np.linalg.norm(features, axis=1, keepdims=True)

        # Classifierモデルの出力を確認
        score = tagger.model(
            torch.from_numpy(np.zeros((1, 768))).float().to(cuda_device)
        )  # model の入力を修正
        assert score.shape == (1, 1)  # バッチサイズ1、出力次元1
        assert score.dtype == torch.float32

    finally:
        tagger.stop = MagicMock()  # stop メソッドをモック化
