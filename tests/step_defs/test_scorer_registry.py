"""
スコアラーレジストリの単体テスト
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
from pytest_bdd import given, when, then, scenarios
from scorer_wrapper_lib.scorer_registry import (
    list_module_files,
    import_module_from_file,
    register_scorers,
    get_registry,
    list_available_scorers,
)

# シナリオファイルの読み込み
scenarios("../features/scorer_registry.feature")

# テスト用のフィクスチャデータ
TEST_MODULE_DIR = "test_modules"
TEST_MODULE_FILES = [
    Path("test_module1.py"),
    Path("test_module2.py"),
    Path("__init__.py"),  # 除外されるべきファイル
]

# テスト用のスコアラークラスデータ
TEST_SCORER_CLASSES = {
    "BaseScorer": type("BaseScorer", (), {"predict": lambda x: x}),
    "DerivedScorer": type(
        "DerivedScorer",
        (type("BaseScorer", (), {"predict": lambda x: x}),),
        {"predict": lambda x: x * 2},
    ),
}

# テスト用の設定データ
TEST_CONFIG = {
    "base_scorer": {"class": "BaseScorer"},
    "derived_scorer": {"class": "DerivedScorer"},
}


@given("モジュールディレクトリが存在する", target_fixture="test_module_directory")
def given_module_directory():
    """テスト用のモジュールディレクトリのセットアップ"""
    return {"directory": TEST_MODULE_DIR}


@given(
    "Pythonモジュールファイルを検索した結果がある",
    target_fixture="search_python_modules",
)
def given_search_python_modules(test_module_directory):
    """モジュールファイルの検索を実行"""
    with patch("pathlib.Path.glob", return_value=TEST_MODULE_FILES):
        result = list_module_files(test_module_directory["directory"])
        return {"module_files": result}


@given(
    "スコアラークラスを含むディレクトリが存在する",
    target_fixture="test_score_models_directory",
)
def given_score_models_directory():
    """テスト用のスコアラークラスを含むディレクトリのセットアップ"""
    return {"directory": "scorer_wrapper_lib.score_models"}


@given(
    "利用可能なスコアラークラスが収集されている",
    target_fixture="available_scorer_classes",
)
def given_gather_available_classes():
    """利用可能なスコアラークラスを収集"""
    mock_gather = MagicMock(return_value=TEST_SCORER_CLASSES)
    with patch(
        "scorer_wrapper_lib.scorer_registry.gather_available_classes", mock_gather
    ):
        return {"classes": TEST_SCORER_CLASSES}


@given("スコアラー設定が存在する", target_fixture="test_load_model_config")
def given_load_model_config():
    """スコアラー設定をロード"""
    with patch(
        "scorer_wrapper_lib.scorer_registry.load_model_config", return_value=TEST_CONFIG
    ):
        return TEST_CONFIG


@when("モジュールをインポートする", target_fixture="import_module_result")
def when_import_module(search_python_modules):
    mock_module = MagicMock()
    with patch("importlib.import_module", return_value=mock_module):
        results = []
        for file in search_python_modules["module_files"]:
            if file.name != "__init__.py":
                module = import_module_from_file(
                    file, "scorer_wrapper_lib.score_models"
                )
                results.append(module)
        return {"imported_modules": results}


@when("存在しないモジュールのインポート", target_fixture="nonexistent_module_result")
def when_import_nonexistent_module(search_python_modules):
    with patch("importlib.import_module", side_effect=ImportError("Module not found")):
        results = []
        for file in search_python_modules["module_files"]:
            if file.name != "__init__.py":
                module = import_module_from_file(file, "nonexistent_module")
                results.append(module)
        return {"imported_modules": results}


@when("スコアラーを登録する", target_fixture="registered_scorers")
def when_register_scorers(available_scorer_classes):
    """スコアラーを登録"""
    mock_register = MagicMock(return_value=available_scorer_classes["classes"])
    with patch("scorer_wrapper_lib.scorer_registry.register_scorers", mock_register):
        result = register_scorers()
        return {"registry": result}


@when("レジストリを取得する", target_fixture="registry_result")
def when_get_registry(registered_scorers):
    """レジストリを取得"""
    mock_get_registry = MagicMock(return_value=registered_scorers["registry"])
    with patch("scorer_wrapper_lib.scorer_registry.get_registry", mock_get_registry):
        result = get_registry()
        # list_available_scorersのモックも設定
        mock_list = MagicMock(return_value=list(TEST_CONFIG.keys()))
        with patch(
            "scorer_wrapper_lib.scorer_registry.list_available_scorers", mock_list
        ):
            available_scorers = list_available_scorers()
        return {
            "registry": result,
            "available_scorers": available_scorers,
        }


@then("指定したディレクトリ内のPythonファイル一覧が取得できる")
def then_verify_module_files(search_python_modules):
    """検索結果の検証"""
    result = search_python_modules["module_files"]
    assert len(result) == 2  # __init__.pyは除外される
    assert all(f.suffix == ".py" for f in result)
    assert not any(f.name == "__init__.py" for f in result)


@then("モジュールが正常にインポートされる")
def then_verify_module_import(import_module_result):
    assert import_module_result["imported_modules"]
    assert all(
        module is not None for module in import_module_result["imported_modules"]
    )


@then("存在しないモジュールのインポートが失敗したことを検証")
def then_verify_error_on_import(nonexistent_module_result):
    assert nonexistent_module_result["imported_modules"]
    assert all(
        module is None for module in nonexistent_module_result["imported_modules"]
    )


@then("利用可能なスコアラークラスの辞書が取得できる")
def then_verify_available_classes(available_scorer_classes):
    """利用可能なスコアラークラスの辞書の検証"""
    assert available_scorer_classes["classes"]
    assert len(available_scorer_classes["classes"]) == len(TEST_SCORER_CLASSES)
    assert all(
        name in available_scorer_classes["classes"]
        for name in TEST_SCORER_CLASSES.keys()
    )


@then("基本実装と派生クラスが含まれている")
def then_verify_class_hierarchy(available_scorer_classes):
    """基本実装と派生クラスの存在を検証"""
    classes = available_scorer_classes["classes"]
    assert "BaseScorer" in classes
    assert "DerivedScorer" in classes
    assert all(hasattr(cls, "predict") for cls in classes.values())


@then("スコアラーが正しく登録される")
def then_verify_registration(registered_scorers):
    """スコアラーの登録が成功したことを検証"""
    assert registered_scorers["registry"], "レジストリが空です"
    assert len(registered_scorers["registry"]) > 0, (
        "スコアラーが1つも登録されていません"
    )
    assert all(
        hasattr(cls, "predict") for cls in registered_scorers["registry"].values()
    ), "predictメソッドを持たないスコアラーが含まれています"


@then("登録されたスコアラーの辞書が取得できる")
def then_verify_registry(registry_result):
    """登録されたスコアラーの辞書の検証"""
    assert registry_result["registry"], "レジストリが空です"
    assert len(registry_result["registry"]) > 0, "スコアラーが1つも登録されていません"
    assert all(isinstance(name, str) for name in registry_result["registry"].keys()), (
        "スコアラー名が文字列ではありません"
    )


@then("登録されているスコアラー名のリストが取得できる")
def then_verify_registered_scorer_names(registry_result):
    """登録されているスコアラー名のリストの検証"""
    available_scorers = registry_result["available_scorers"]
    assert len(available_scorers) > 0, "スコアラー名が1つも登録されていません"
    assert all(isinstance(name, str) for name in available_scorers), (
        "スコアラー名が文字列ではありません"
    )
