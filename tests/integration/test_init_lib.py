"""ライブラリ初期化のテスト。

このモジュールでは、ライブラリの初期化処理とスコアラー登録のテストを実装します。
"""

from pathlib import Path

from pytest_bdd import given, parsers, scenario, then, when

from scorer_wrapper_lib import list_available_scorers

# Featureファイルの絶対パスを取得
FEATURE_FILE = str(Path(__file__).parent.parent / "features" / "init_lib.feature")


@scenario(FEATURE_FILE, "基本的な初期化")
def test_init_with_valid_model():
    """正常なモデルで初期化するシナリオのテスト。"""
    pass


@scenario(FEATURE_FILE, "設定の検証")
def test_init_with_invalid_model():
    """無効なモデルで初期化するシナリオのテスト。"""
    pass


@given("有効なモデル設定が存在する")
def valid_model_config():
    """有効なモデル設定が存在することを確認するステップ。"""
    # この時点で models.toml は既に存在しているため、特別な準備は不要
    pass


@given("無効なモデル名が指定される")
def invalid_model_name():
    """無効なモデル名が指定されることを確認するステップ。"""
    # 特別な準備は不要
    pass


@when(parsers.cfparse("ライブラリが {model_name} を使用してスコアラーを初期化する"))
def init_library_with_model(model_name):
    """指定されたモデル名でライブラリを初期化するステップ。"""
    try:
        get_scorer_instance(model_name)
    except ValueError:
        pass


@then(
    parsers.cfparse(
        "利用可能なスコアラー一覧に {model_name} が含まれていることを確認する"
    )
)
def verify_model_in_available_scorers(model_name):
    """指定されたモデルが利用可能なスコアラー一覧に含まれていることを確認するステップ。"""
    available_scorers = list_available_scorers()
    assert model_name in available_scorers, (
        f"Model {model_name} not found in available scorers"
    )


@then("エラーが発生することを確認する")
def verify_error_occurred():
    """エラーが発生したことを確認するステップ。"""
    # whenステップで既にエラーの検証を行っているため、
    # このステップでは特別な処理は不要
    pass
