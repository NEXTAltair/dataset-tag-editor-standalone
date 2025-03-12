"""スコアラーレジストリモジュールの統合テスト

このモジュールでは、スコアラーレジストリモジュールの統合テストを実装します。
"""

import pytest
from pathlib import Path

from pytest_bdd import given, scenario, then, when

# Featureファイルの絶対パスを取得
FEATURE_FILE = str(
    Path(__file__).parent.parent / "features" / "scorer_registry.feature"
)


# シナリオ定義
@scenario(FEATURE_FILE, "スコアラーの登録")
def test_スコアラーの登録():
    """スコアラー登録のシナリオテスト。"""
    pass


@scenario(FEATURE_FILE, "レジストリの取得")
def test_登録済みスコアラーの取得():
    """登録済みスコアラー取得のシナリオテスト。"""
    pass


@scenario(FEATURE_FILE, "無効なディレクトリからのモジュールファイルリスト取得")
def test_未登録スコアラーの取得():
    """未登録スコアラー取得のシナリオテスト。"""
    pass


@scenario(FEATURE_FILE, "利用可能なスコアラーの一覧取得")
def test_登録済みスコアラーのリスト取得():
    """登録済みスコアラーリスト取得のシナリオテスト。"""
    pass


@scenario(FEATURE_FILE, "無効なファイルからのモジュールインポート")
def test_無効なファイルからのモジュールインポート():
    """無効なファイルからのモジュールインポートのシナリオテスト。"""
    pass


# 共有変数
registry = None
scorers_list = None
module_files = None
import_result = None


# スコアラーの登録テスト用ステップ
@given("テスト用のスコアラー設定が存在する")
def given_scorer_config_exists():
    # 実際のテスト環境ではシステムに設定が存在していることを前提とする
    # 統合テストなのでモックは使わない
    pass


@when("register_scorers関数を呼び出す")
def when_register_scorers_called():
    try:
        from scorer_wrapper_lib.scorer_registry import register_scorers

        global registry
        registry = register_scorers()
    except ImportError:
        pytest.skip("scorer_wrapper_lib をインポートできませんでした")


@then("スコアラーが正しく登録される")
def then_scorers_registered():
    assert registry is not None
    assert isinstance(registry, dict)
    assert len(registry) > 0


# レジストリの取得テスト用ステップ
@given("スコアラーが登録されている")
def given_scorers_registered():
    try:
        from scorer_wrapper_lib.scorer_registry import register_scorers

        global registry
        registry = register_scorers()
        assert isinstance(registry, dict), "レジストリが正しく取得できませんでした"
    except ImportError:
        pytest.skip("scorer_wrapper_lib をインポートできませんでした")


@when("get_registry関数を呼び出す")
def when_get_registry_called():
    try:
        from scorer_wrapper_lib.scorer_registry import get_registry

        global registry
        registry = get_registry()
    except ImportError:
        pytest.skip("scorer_wrapper_lib をインポートできませんでした")


@then("登録されたスコアラーの辞書が取得できる")
def then_registry_obtained():
    assert registry is not None
    assert isinstance(registry, dict)
    assert len(registry) > 0


# スコアラーの一覧取得テスト用ステップ
@given("複数のスコアラーが登録されている")
def given_multiple_scorers_registered():
    try:
        from scorer_wrapper_lib.scorer_registry import register_scorers

        register_scorers()
    except ImportError:
        pytest.skip("scorer_wrapper_lib をインポートできませんでした")


@when("list_available_scorers関数を呼び出す")
def when_list_available_scorers_called():
    try:
        from scorer_wrapper_lib.scorer_registry import list_available_scorers

        global scorers_list
        scorers_list = list_available_scorers()
    except ImportError:
        pytest.skip("scorer_wrapper_lib をインポートできませんでした")


@then("登録されているスコアラー名のリストが取得できる")
def then_scorer_names_list_obtained():
    assert scorers_list is not None
    assert isinstance(scorers_list, list)


# 無効なディレクトリからのファイルリスト取得テスト用ステップ
@given("存在しないディレクトリが指定される")
def given_nonexistent_directory_specified():
    # 存在しないディレクトリのパスを設定
    global nonexistent_dir
    nonexistent_dir = "/this/directory/does/not/exist"


@when("list_module_files関数を呼び出す")
def when_list_module_files_called():
    try:
        from scorer_wrapper_lib.scorer_registry import list_module_files

        global module_files
        module_files = list_module_files(nonexistent_dir)
    except ImportError:
        pytest.skip("scorer_wrapper_lib をインポートできませんでした")


@then("空のリストが返される")
def then_empty_list_returned():
    assert module_files == []


# 無効なファイルからのモジュールインポートテスト用ステップ
@given("存在しないモジュールファイルが指定される")
def given_nonexistent_file_specified():
    # 存在しないファイルのパスを設定
    global nonexistent_file
    nonexistent_file = Path("/this/file/does/not/exist.py")


@when("import_module_from_file関数を呼び出す")
def when_import_module_from_file_called():
    try:
        from scorer_wrapper_lib.scorer_registry import import_module_from_file

        global import_result
        import_result = import_module_from_file(nonexistent_file, "test_module")
    except ImportError:
        pytest.skip("scorer_wrapper_lib をインポートできませんでした")


@then("適切なエラーが発生する")
def then_appropriate_error_occurs():
    # import_module_from_file関数はエラーをキャッチしてNoneを返すため、
    # 結果がNoneであることを確認する
    assert import_result is None
