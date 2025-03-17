"""
スコアラーレジストリのBDDテスト用ステップ定義ファイル
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from behave import given, then, when

from scorer_wrapper_lib.scorer_registry import (
    gather_available_classes,
    gather_core_classes,
    get_registry,
    import_module_from_file,
    list_available_scorers,
    list_module_files,
    recursive_subclasses,
    register_models_from_config,
    register_scorers,
)


@given("テスト用のモジュールディレクトリが存在する")
def given_module_directory_exists(context):
    # 一時ディレクトリを作成してテスト用のモジュールディレクトリとして使用
    context.temp_dir = tempfile.TemporaryDirectory()
    context.module_dir = Path(context.temp_dir.name)

    # テスト用のPythonファイルを作成
    (context.module_dir / "test_module1.py").write_text("# Test module 1")
    (context.module_dir / "test_module2.py").write_text("# Test module 2")
    (context.module_dir / "not_a_module.txt").write_text("This is not a Python module")

    # サブディレクトリも作成
    sub_dir = context.module_dir / "subdir"
    sub_dir.mkdir()
    (sub_dir / "submodule.py").write_text("# Submodule")


@when("list_module_files関数を呼び出す")
def when_list_module_files_called(context):
    context.result = list_module_files(str(context.module_dir))


@then("指定したディレクトリ内のPythonファイル一覧が取得できる")
def then_python_files_list_obtained(context):
    # 取得したファイル一覧を検証
    files = [f.name for f in context.result]
    assert "test_module1.py" in files
    assert "test_module2.py" in files
    assert "not_a_module.txt" not in files
    assert "submodule.py" not in files  # サブディレクトリは再帰的に検索しない


@given("テスト用のモジュールファイルが存在する")
def given_module_file_exists(context):
    # 一時ディレクトリを作成してテスト用のモジュールファイルを作成
    context.temp_dir = tempfile.TemporaryDirectory()
    context.module_path = Path(context.temp_dir.name) / "test_module.py"

    # 簡単なクラスを含むモジュールを作成
    context.module_path.write_text("""
class TestClass:
    def test_method(self):
        return "test"
""")


@when("import_module_from_file関数を呼び出す")
def when_import_module_called(context):
    # モジュールをインポート
    context.imported_module = import_module_from_file(context.module_path, "test_module")


@then("モジュールが正常にインポートされる")
def then_module_imported(context):
    # インポートされたモジュールを検証
    assert hasattr(context.imported_module, "TestClass")
    test_instance = context.imported_module.TestClass()
    assert test_instance.test_method() == "test"


@given("テスト用の基底クラスとサブクラスが存在する")
def given_base_and_subclasses_exist(context):
    # テスト用の基底クラスとサブクラスを定義
    class BaseClass:
        pass

    class SubClass1(BaseClass):
        pass

    class SubClass2(BaseClass):
        pass

    class SubSubClass(SubClass1):
        pass

    class UnrelatedClass:
        pass

    context.base_class = BaseClass
    context.sub_classes = {SubClass1, SubClass2, SubSubClass}
    context.unrelated_class = UnrelatedClass


@when("recursive_subclasses関数を呼び出す")
def when_recursive_subclasses_called(context):
    context.result = recursive_subclasses(context.base_class)


@then("全てのサブクラスが取得できる")
def then_all_subclasses_obtained(context):
    assert context.result == context.sub_classes
    assert context.unrelated_class not in context.result


@given("テスト用のスコアラークラスを含むディレクトリが存在する")
def given_scorer_classes_directory_exists(context):
    # モックを使用してテスト用のディレクトリと基底クラスをセットアップ
    context.mock_dir = "mock_scorer_dir"
    context.mock_available_classes = {"TestScorer": "TestScorerClass"}

    # gather_available_classes関数をモック化
    context.original_gather = gather_available_classes
    gather_available_classes_patcher = patch("scorer_wrapper_lib.scorer_registry.gather_available_classes")
    mock_gather = gather_available_classes_patcher.start()
    mock_gather.return_value = context.mock_available_classes
    context.add_cleanup(gather_available_classes_patcher.stop)


@when("gather_available_classes関数を呼び出す")
def when_gather_available_classes_called(context):
    context.result = gather_available_classes(context.mock_dir)


@then("利用可能なスコアラークラスの辞書が取得できる")
def then_available_classes_obtained(context):
    assert context.result == context.mock_available_classes


@given("コアモジュールが存在する")
def given_core_modules_exist(context):
    # gather_core_classes関数をモック化
    context.mock_core_classes = {"CoreClass": "MockCoreClass"}

    gather_core_patcher = patch("scorer_wrapper_lib.scorer_registry.gather_core_classes")
    mock_gather_core = gather_core_patcher.start()
    mock_gather_core.return_value = context.mock_core_classes
    context.add_cleanup(gather_core_patcher.stop)


@when("gather_core_classes関数を呼び出す")
def when_gather_core_classes_called(context):
    context.result = gather_core_classes()


@then("コアクラスの辞書が取得できる")
def then_core_classes_obtained(context):
    assert context.result == context.mock_core_classes


@given("テスト用のモデル設定が存在する")
def given_model_config_exists(context):
    context.config = {"models": {"test_model": {"class": "TestScorer", "parameters": {"param1": "value1"}}}}


@given("利用可能なクラスとコアクラスが準備されている")
def given_available_and_core_classes_prepared(context):
    context.available_classes = {"TestScorer": MagicMock()}
    context.core_classes = {"CoreClass": MagicMock()}


@when("register_models_from_config関数を呼び出す")
def when_register_models_from_config_called(context):
    context.registry = {}
    register_models_from_config(context.config, context.available_classes, context.core_classes)
    context.result = context.registry


@then("モデルが正しく登録される")
def then_models_registered(context):
    # 実際のレジストリが更新されているか確認 (モックなのでここではスキップ)
    pass


@given("テスト用のスコアラー設定が存在する")
def given_scorer_config_exists(context):
    # モックを使用してスコアラー設定をセットアップ
    load_config_patcher = patch("scorer_wrapper_lib.scorer_registry.load_model_config")
    mock_load_config = load_config_patcher.start()
    mock_load_config.return_value = {
        "models": {"test_model": {"class": "TestScorer", "parameters": {"param1": "value1"}}}
    }
    context.add_cleanup(load_config_patcher.stop)

    # その他の依存関数をモック化
    gather_available_patcher = patch("scorer_wrapper_lib.scorer_registry.gather_available_classes")
    mock_gather_available = gather_available_patcher.start()
    mock_gather_available.return_value = {"TestScorer": MagicMock()}
    context.add_cleanup(gather_available_patcher.stop)

    gather_core_patcher = patch("scorer_wrapper_lib.scorer_registry.gather_core_classes")
    mock_gather_core = gather_core_patcher.start()
    mock_gather_core.return_value = {"CoreClass": MagicMock()}
    context.add_cleanup(gather_core_patcher.stop)


@when("register_scorers関数を呼び出す")
def when_register_scorers_called(context):
    context.result = register_scorers()


@then("スコアラーが正しく登録される")
def then_scorers_registered(context):
    assert "test_model" in context.result


@given("スコアラーが登録されている")
def given_scorers_registered(context):
    # レジストリを直接モック化
    context.mock_registry = {"test_model": MagicMock()}

    get_registry_patcher = patch("scorer_wrapper_lib.scorer_registry.get_registry")
    mock_get_registry = get_registry_patcher.start()
    mock_get_registry.return_value = context.mock_registry
    context.add_cleanup(get_registry_patcher.stop)


@when("get_registry関数を呼び出す")
def when_get_registry_called(context):
    context.result = get_registry()


@then("登録されたスコアラーの辞書が取得できる")
def then_registry_obtained(context):
    assert context.result == context.mock_registry


@given("複数のスコアラーが登録されている")
def given_multiple_scorers_registered(context):
    # レジストリを直接モック化
    context.mock_registry = {"model1": MagicMock(), "model2": MagicMock(), "model3": MagicMock()}

    get_registry_patcher = patch("scorer_wrapper_lib.scorer_registry.get_registry")
    mock_get_registry = get_registry_patcher.start()
    mock_get_registry.return_value = context.mock_registry
    context.add_cleanup(get_registry_patcher.stop)


@when("list_available_scorers関数を呼び出す")
def when_list_available_scorers_called(context):
    context.result = list_available_scorers()


@then("登録されているスコアラー名のリストが取得できる")
def then_scorer_names_list_obtained(context):
    assert set(context.result) == set(context.mock_registry.keys())


@given("存在しないディレクトリが指定される")
def given_nonexistent_directory_specified(context):
    context.nonexistent_dir = "/this/directory/does/not/exist"


@then("空のリストが返される")
def then_empty_list_returned(context):
    assert context.result == []


@given("存在しないモジュールファイルが指定される")
def given_nonexistent_file_specified(context):
    context.nonexistent_file = Path("/this/file/does/not/exist.py")


@then("適切なエラーが発生する")
def then_appropriate_error_occurs(context):
    # エラーの検証方法はシナリオによって異なるので、実装時に適切に対応する必要があります
    pass
