"""スコアラーレジストリモジュールの統合テスト

このモジュールでは、スコアラーレジストリモジュールの統合テストを実装します。
"""

from pytest_bdd import given, scenarios, then, when

from image_annotator_lib.core.config import config_registry
from image_annotator_lib.core.registry import (  # type: ignore
    get_cls_obj_registry,
    list_available_annotators,
    register_annotators,
)

scenarios("../features/core/registry.feature")


@given("モデル設定TOMLファイルが存在する", target_fixture="test_config_toml")
def given_scorer_config_exists():
    config_registry.load()
    return config_registry.get_all_config()


@given(
    "モデルモジュールディレクトリが利用可能である",
    target_fixture="test_module_directory",
)
def given_module_directory_exists():
    # 統合テストパッケージとしてインストールした
    # image_annotator_lib/score_modelsディレクトリが存在していることを前提とする
    return "score_models"


@given("モデルレジストリが構築されている", target_fixture="test_registry")
def given_model_registry_built():
    return register_annotators()


@when("モデルクラスオブジェクトレジストリを構築する", target_fixture="test_registry")
def whem_model_registry_built():
    return register_annotators()


@when("利用可能なモデル名のリストを取得する", target_fixture="test_model_name_list")
def when_available_model_names_list_obtained():
    return list_available_annotators()


@when("レジストリから特定のモデルを取得する", target_fixture="test_specific_model")
def when_specific_model_obtained_from_registry():
    registry = get_cls_obj_registry()
    # レジストリから最初のモデルを取得
    if registry:
        model_name = next(iter(registry.keys()))
        specific_model = registry[model_name]
        return specific_model


@then("モデル名をキーにモデルクラスオブジェクトが登録されている")
def then_model_name_key_model_class_object_registered(test_registry):
    # レジストリが空でないことを確認
    assert len(test_registry) > 0, "レジストリにモデルが登録されていません"

    # 各値がクラスオブジェクト（type型）であることを確認
    for model_name, model_class in test_registry.items():
        assert isinstance(model_class, type), (
            f"値はクラスオブジェクトである必要があります: {model_name} -> {model_class}"
        )


@then("レジストリの内容は設定ファイルの内容と一致する")
def then_registry_content_matches_config_file(test_config_toml, test_registry):
    for model_name, model_config in test_config_toml.items():
        assert model_name in test_registry.keys(), (
            f"config_toml に存在するモデル: {model_name} がレジストリに存在しません"
        )

        # クラス名で比較（レジストリのクラスオブジェクトから__name__を取得）
        class_name = test_registry[model_name].__name__
        assert class_name == model_config["class"], (
            f"モデル {model_name} のクラス名が一致しません: "
            f"レジストリには {class_name}, 設定には {model_config['class']}"
        )


@then("設定ファイルに記述されたすべてのモデル名がリストに含まれている")
def then_all_model_names_in_config_file_in_list(test_model_name_list, test_config_toml):
    assert set(test_model_name_list) == set(test_config_toml.keys()), (
        "設定ファイルに記述されたすべてのモデル名がリストに含まれていません"
    )


@then("モデル名に対応するクラスオブジェクトが返される")
def then_model_name_corresponding_class_object_returned(test_specific_model):
    # 取得したクラスオブジェクトがNoneでないことを確認
    assert test_specific_model is not None, "モデルが取得できませんでした"

    # 取得したオブジェクトがクラス（type）であることを確認
    assert isinstance(test_specific_model, type), (
        f"取得したオブジェクトはクラスオブジェクトではありません: {type(test_specific_model)}"
    )

    # クラスオブジェクトの名前が意味のある値であることを確認（オプション）
    class_name = test_specific_model.__name__
    assert len(class_name) > 0, "クラス名が空です"
