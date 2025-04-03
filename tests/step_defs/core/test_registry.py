"""
スコアラーレジストリの単体テスト
"""

import pytest

from unittest.mock import patch
from pytest_bdd import given, when, then, scenarios
from image_annotator_lib.core.registry import (
    get_cls_obj_registry,
    register_annotators,
)

# シナリオファイルの読み込み
scenarios("../../features/core/registry.feature")

# テスト用のフィクスチャデータ
TEST_MODULE_DIR = "test_modules"


@given("モデル設定TOMLファイルが存在する", target_fixture="test_config_toml")
def given_config_toml_exists(mock_config_toml):
    """モデル設定TOMLファイルのセットアップ"""
    return mock_config_toml


@given(
    "モデルモジュールディレクトリが利用可能である",
    target_fixture="test_module_directory",
)
def given_module_directory_exists():
    """モジュールディレクトリのセットアップ"""
    return TEST_MODULE_DIR


@given("モデルレジストリが構築されている", target_fixture="test_registry")
def given_model_registry_built(test_config_toml):
    """テスト用のレジストリを構築"""

    # テスト用のモデルクラス定義
    mock_classes = {
        "TestScorer01": type("TestScorer01", (), {"predict": lambda x: x}),
        "TestScorer02": type("TestScorer02", (), {"predict": lambda x: x * 2}),
    }

    # register_annotatorsの中で使用される依存関係をモック化
    with patch(
        "image_annotator_lib.core.registry.load_model_config",
        return_value=test_config_toml,
    ):
        with patch(
            "image_annotator_lib.core.registry._gather_available_classes",
            return_value=mock_classes,
        ):
            registry = register_annotators()
            return registry


@when("モデルクラスオブジェクトレジストリを構築する", target_fixture="test_registry")
def when_model_registry_built(test_config_toml):
    return given_model_registry_built(test_config_toml)


@when("利用可能なモデル名のリストを取得する", target_fixture="test_model_name_list")
def when_available_model_names_list_obtained(test_registry):
    # デバッグ情報
    print("\n=== デバッグ情報 ===")
    print(f"テストレジストリのキー: {list(test_registry.keys())}")

    from image_annotator_lib.core.registry import _MODEL_CLASS_OBJ_REGISTRY

    print(f"グローバルレジストリのキー: {list(_MODEL_CLASS_OBJ_REGISTRY.keys())}")
    print("=== デバッグ情報終了 ===\n")

    # テスト用レジストリからモデル名を直接取得
    return list(test_registry.keys())


@when("レジストリから特定のモデルを取得する", target_fixture="test_specific_model")
def when_specific_model_obtained_from_registry(test_registry):
    registry = get_cls_obj_registry()
    # レジストリから最初のモデルを取得
    if registry:
        model_name = next(iter(registry.keys()))
        specific_model = registry[model_name]
        return specific_model
    else:
        pytest.fail("レジストリにモデルが存在しません")


@then("モデル名をキーにモデルクラスオブジェクトが登録されている")
def then_model_name_key_model_class_object_registered(test_registry):
    # レジストリが空でないことを確認
    assert len(test_registry) > 0, "レジストリにモデルが登録されていません"

    # 各値がクラスオブジェクト（type型）であることを確認
    for model_name, model_class in test_registry.items():
        assert isinstance(
            model_class, type
        ), f"値はクラスオブジェクトである必要があります: {model_name} -> {model_class}"


@then("レジストリの内容は設定ファイルの内容と一致する")
def then_registry_content_matches_config_file(test_config_toml, test_registry):
    for model_name, model_config in test_config_toml.items():
        assert (
            model_name in test_registry.keys()
        ), f"config_toml に存在するモデル: {model_name} がレジストリに存在しません"

        # クラス名で比較（レジストリのクラスオブジェクトから__name__を取得）
        class_name = test_registry[model_name].__name__
        assert class_name == model_config["class"], (
            f"モデル {model_name} のクラス名が一致しません: "
            f"レジストリには {class_name}, 設定には {model_config['class']}"
        )


@then("設定ファイルに記述されたすべてのモデル名がリストに含まれている")
def then_all_model_names_in_config_file_in_list(test_model_name_list, test_config_toml):
    assert set(test_config_toml.keys()).issubset(set(test_model_name_list))


@then("モデル名に対応するクラスオブジェクトが返される")
def then_model_name_corresponding_class_object_returned(test_specific_model):
    # 取得したクラスオブジェクトがNoneでないことを確認
    assert test_specific_model is not None, "モデルが取得できませんでした"

    # 取得したオブジェクトがクラス（type）であることを確認
    assert isinstance(
        test_specific_model, type
    ), f"取得したオブジェクトはクラスオブジェクトではありません: {type(test_specific_model)}"

    # クラスオブジェクトの名前が意味のある値であることを確認（オプション）
    class_name = test_specific_model.__name__
    assert len(class_name) > 0, "クラス名が空です"
