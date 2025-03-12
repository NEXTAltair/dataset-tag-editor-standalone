from pathlib import Path

import pytest
from PIL import Image
from pytest_bdd import given, parsers, scenario, then, when

# グローバル変数
scorer = None
another_scorer = None
second_scorer = None
test_images = []
evaluation_results = None

# Featureファイルの絶対パスを取得
FEATURE_FILE = str(Path(__file__).parent / "features" / "cache_handling.feature")


@scenario(FEATURE_FILE, "モデルのキャッシュと復元")
def test_cache_handling():
    pass


@scenario(FEATURE_FILE, "複数モデルの初期化とキャッシュ")
def test_multiple_models_cache():
    pass


@scenario(FEATURE_FILE, "キャッシュからの復元")
def test_restore():
    pass


@scenario(FEATURE_FILE, "無効なモデル名でのエラー処理")
def test_invalid_model_error():
    pass


@scenario(FEATURE_FILE, "キャッシュと解放の繰り返しによるメモリ管理")
def test_cache_release_cycle():
    pass


@given("有効なモデル設定が存在する")
def _():
    pass


@given("テスト用の画像が準備されている")
def prepare_test_images():
    global test_images
    # テスト用の空の画像を作成
    test_images = [Image.new("RGB", (100, 100), color="white")]


@when(parsers.parse('ライブラリが "{model_name}" を使用してスコアラーを初期化する'))
def init_model(model_name):
    global scorer
    scorer = get_scorer_instance(model_name)


@when(parsers.parse('ライブラリが "{model_name}" を使用して別のスコアラーを初期化する'))
def init_another_model(model_name):
    global another_scorer
    another_scorer = get_scorer_instance(model_name)


@when(parsers.parse('再度ライブラリが "{model_name}" を使用して別のスコアラーを初期化する'))
def init_second_model(model_name):
    global second_scorer
    second_scorer = get_scorer_instance(model_name)


@when("スコアラーの load_or_restore_model メソッドを呼び出す")
def load_model():
    global scorer
    scorer.load_or_restore_model()


@when("両方のスコアラーの load_or_restore_model メソッドを呼び出す")
def load_both_models():
    global scorer, another_scorer
    scorer.load_or_restore_model()
    another_scorer.load_or_restore_model()


@when("2つ目のスコアラーの load_or_restore_model メソッドを呼び出す")
def load_second_model():
    global second_scorer
    second_scorer.load_or_restore_model()


@when("スコアラーの cache_to_main_memory メソッドを呼び出す")
def cache_model():
    global scorer
    scorer.cache_to_main_memory()


@when("両方のスコアラーの cache_to_main_memory メソッドを呼び出す")
def cache_both_models():
    global scorer, another_scorer
    scorer.cache_to_main_memory()
    another_scorer.cache_to_main_memory()


@when("スコアラーの restore_from_main_memory メソッドを呼び出す")
def restore_model():
    global scorer
    scorer.restore_from_main_memory()


@when("スコアラーの release_model メソッドを呼び出す")
def release_model():
    global scorer
    scorer.release_model()


@when("再度スコアラーの cache_to_main_memory メソッドを呼び出す")
def cache_model_again():
    global scorer
    scorer.cache_to_main_memory()


@when("再度スコアラーの release_model メソッドを呼び出す")
def release_model_again():
    global scorer
    scorer.release_model()


@when(parsers.parse('ライブラリが "{model_name}" を使用してスコアラーを初期化しようとする'))
def try_init_invalid_model(model_name):
    global scorer
    try:
        scorer = get_scorer_instance(model_name)
    except ValueError as e:
        # エラーを捕捉して後で検証できるように保存
        pytest.exception = e


@then("スコアラーのモデルが None でないことを確認する")
def check_model_not_none():
    global scorer
    assert scorer.model is not None


@then("スコアラーのモデルが正常に復元されていることを確認する")
def check_model_restored():
    global scorer
    assert scorer.model is not None
    # 必要に応じて復元されたモデルの状態を検証


@then("両方のスコアラーのモデルが正常にキャッシュされていることを確認する")
def check_both_models_cached():
    global scorer, another_scorer
    assert scorer.is_cached()
    assert another_scorer.is_cached()


@then("適切なエラーが発生することを確認する")
def check_error_raised():
    assert hasattr(pytest, "exception")
    assert isinstance(pytest.exception, ValueError)
    assert "Model config not found" in str(pytest.exception)


@then("システムリソースが適切に管理されていることを確認する")
def check_system_resources():
    global scorer, second_scorer
    # 最初のスコアラーのモデルが解放されていることを確認
    assert scorer.model is None
    # 次のスコアラーのモデルがロードされていることを確認
    assert second_scorer.model is not None
    # メモリリークがないことを確認（必要に応じて）
