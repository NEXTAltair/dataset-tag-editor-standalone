"""
スコアラーのBDDテスト用ステップ定義ファイル
"""

from unittest.mock import MagicMock, patch
import pytest
from pytest_bdd import given, then, when, parsers, scenarios

from image_annotator_lib.api import (
    _MODEL_INSTANCE_REGISTRY,
    _create_annotator_instance,
    get_annotator_instance,
    annotate,
)

scenarios("../features/scorer.feature")


@pytest.fixture
def mock_registry():
    """get_cls_obj_registry"""
    registry = {}

    # get_cls_obj_registryのモック
    patcher = patch("image_annotator_lib.core.registry.get_cls_obj_registry")
    mock_get_cls_obj_registry = patcher.start()
    mock_get_cls_obj_registry.return_value = registry

    def register(model_name, predict_results=None):
        """モデルの登録"""
        # モデルクラスのモックを作成
        mock_class = MagicMock()
        # インスタンスのモックを作成
        mock_instance = MagicMock()
        mock_instance.model_name = model_name
        if predict_results:
            mock_instance.predict.return_value = predict_results

        # クラスのモックがインスタンス化されたときにmock_instanceを返すように設定
        mock_class.return_value = mock_instance
        # クラスのモックをregistryに登録
        registry[model_name] = mock_class
        return mock_instance

    # registryとregister関数の両方を返す
    yield register, registry

    patcher.stop()


@given(parsers.parse("{model_name} はまだキャッシュされていない"))
def given_model_not_cached(target_model_list):
    # すでにインポート済みの_MODEL_INSTANCE_REGISTRYを使用
    if target_model_list in _MODEL_INSTANCE_REGISTRY:
        del _MODEL_INSTANCE_REGISTRY[target_model_list]


@given(
    parsers.parse("{model_name} はすでにキャッシュされている"),
    target_fixture="cached_instance",
)
def given_model_already_cached(target_model_list, mock_registry):
    # キャッシュがクリアされていることを確認
    if target_model_list in _MODEL_INSTANCE_REGISTRY:
        del _MODEL_INSTANCE_REGISTRY[target_model_list]

    # モックファクトリからインスタンス作成
    mock_instance = mock_registry(target_model_list)

    # キャッシュに追加
    _MODEL_INSTANCE_REGISTRY[target_model_list] = mock_instance
    return mock_instance


@given("GPUメモリ不足をシミュレートする環境")
def given_gpu_memory_error():
    return {"error_message": "CUDA out of memory"}


@when(
    parsers.parse("レジストリに登録された {model_name} に対応するクラスを取得する"),
    target_fixture="model_class",
)
def when_get_model_class_from_registry(target_model_list, mock_registry):
    model_name = target_model_list[0]
    register, registry = mock_registry  # 使用するものが明確
    mock_instance = register(model_name)
    model_class = registry[model_name]
    return {"class": model_class, "expected_instance": mock_instance}


@when(
    parsers.parse("{model_name} を引数にモデルクラスをインスタンス化する"),
    target_fixture="test_execution_result",
)
def when_instantiate_model_class(model_name, model_class):
    result = _create_annotator_instance(model_name)
    assert result is model_class["expected_instance"], "生成されたインスタンスがモックと一致しない"
    return result


@when(
    "同じモデルで2回アノテータインスタンスを取得する",
    target_fixture="test_execution_result",
)
def when_get_annotator_instance_called_twice(target_model_list, mock_registry):
    """get_annotator_instanceを2回呼び出して、同じインスタンスが返されることを確認"""
    model_name = target_model_list[0]
    register, _ = mock_registry

    # 1回目の呼び出し
    register(model_name)
    first_instance = get_annotator_instance(model_name)

    # 2回目の呼び出し
    second_instance = get_annotator_instance(model_name)

    # 両方のインスタンスを返して検証に使用
    return {
        "first_call": first_instance,
        "second_call": second_instance,
        "model_name": model_name,
    }


@when("各モデルのアノテータインスタンスを取得する", target_fixture="test_execution_result")
def when_get_annotator_instance_for_each_model(target_model_list, mock_registry):
    register, _ = mock_registry
    instances = {}
    for model_name in target_model_list:
        register(model_name)
        instances[model_name] = get_annotator_instance(model_name)
    return instances


@when("画像評価実行", target_fixture="models_result")
def when_annotate_with_multi_models_execution(target_model_list, target_images, mock_registry):
    register, registry = mock_registry

    # 各モデルの基本的なモックを登録し、predictの戻り値を設定
    for model_name in target_model_list:
        # 画像ごとの評価結果を作成
        predict_results = []
        for _ in range(len(target_images)):
            predict_results.append(
                {
                    "model_output": 0.75,  # ダミースコア
                    "model_name": model_name,
                    "score_tag": "good",
                }
            )
        register(model_name, predict_results=predict_results)

    image_count = len(target_images)

    # 評価実行
    result = annotate(target_images, target_model_list)

    return {
        "result": result,
        "model_list": target_model_list,
        "image_count": image_count,
    }


@then("対応したアノテータインスタンスが生成される")
def then_annotator_instance_created(test_execution_result, target_model_list):
    assert test_execution_result is not None
    assert test_execution_result.model_name == target_model_list[0]


@then("同一のインスタンスが返される")
def then_same_instance_returned(test_execution_result):
    assert (
        test_execution_result["first_call"] is test_execution_result["second_call"]
    ), "2回の呼び出しで異なるインスタンスが返された"
    assert isinstance(test_execution_result["first_call"], MagicMock)


@then("各インスタンスは対応するモデル名を持つ")
def then_instance_has_correct_model_name(test_execution_result, target_model_list):
    for model_name in target_model_list:
        assert model_name in test_execution_result, f"モデル {model_name} のインスタンスが見つかりません"
        assert (
            test_execution_result[model_name].model_name == model_name
        ), f"モデル {model_name} のインスタンスが正しいモデル名を持っていません"


@then("両方のモデルがキャッシュされていることを確認する")
def then_both_models_are_cached(initialized_models):
    first_model = initialized_models["first_model"]
    second_model = initialized_models["second_model"]

    assert first_model in _MODEL_INSTANCE_REGISTRY, f"{first_model}がキャッシュされていない"
    assert second_model in _MODEL_INSTANCE_REGISTRY, f"{second_model}がキャッシュされていない"
    assert (
        _MODEL_INSTANCE_REGISTRY[first_model] is initialized_models["first_instance"]
    ), "キャッシュされたインスタンスが一致しない"
    assert (
        _MODEL_INSTANCE_REGISTRY[second_model] is initialized_models["second_instance"]
    ), "キャッシュされたインスタンスが一致しない"


@then(parsers.parse("辞書のキーが {model_name} の各モデル名と一致することを確認する"))
def then_multi_models_keys_match_model_names(target_model_list, models_result):
    result = models_result["result"]

    for model in target_model_list:
        assert model in result, f"モデル{model}の結果が含まれていない"


@then("各モデル名に対応する値が画像数と同じ長さのリストであることを確認する")
def then_models_result_length_matches_image_count(models_result):
    result = models_result["result"]

    image_count = models_result["image_count"]
    model_list = models_result["model_list"]

    # 各モデルの結果数が画像数と一致
    for model in model_list:
        assert model in result, f"モデル{model}の結果が含まれていない"
        assert (
            len(result[model]) == image_count
        ), f"モデル{model}の結果数({len(result[model])})が画像数({image_count})と一致しない"


# 基本的な機能テスト
def test_create_annotator_instance(mock_registry):
    """アノテータインスタンスの生成テスト"""
    register, registry = mock_registry
    model_name = "test_model"
    register(model_name)  # モデルを登録

    result = _create_annotator_instance(model_name)

    assert result is registry[model_name].return_value


def test_get_annotator_instance_caching(mock_registry):
    """キャッシュ機能のテスト"""
    register, registry = mock_registry
    model_name = "test_model"
    register(model_name)  # モデルを登録

    # 1回目の呼び出し
    first = get_annotator_instance(model_name)
    # 2回目の呼び出し
    second = get_annotator_instance(model_name)

    # 同じインスタンスが返されることを確認
    assert first is second
    # キャッシュに保存されていることを確認
    assert model_name in _MODEL_INSTANCE_REGISTRY
    assert _MODEL_INSTANCE_REGISTRY[model_name] is first
