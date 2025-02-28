from pytest_bdd import given, parsers, scenario, then, when

from scorer_wrapper_lib.scorer import init_scorer


@scenario("features/cache_handling.feature", "モデルのキャッシュと復元")
def test_cache_handling():
    pass


@given("有効なモデル設定が存在する")
def _():
    pass


@when(parsers.parse('ライブラリが "{model_name}" を使用してスコアラーを初期化する'))
def _(model_name):
    global scorer
    scorer = init_scorer(model_name)


@when("スコアラーの load_or_restore_model メソッドを呼び出す")
def _():
    scorer.load_or_restore_model()


@when("スコアラーの cache_to_main_memory メソッドを呼び出す")
def _():
    scorer.cache_to_main_memory()


@when("スコアラーの restore_from_main_memory メソッドを呼び出す")
def _():
    scorer.restore_from_main_memory()


@then("スコアラーのモデルが None でないことを確認する")
def _():
    assert scorer.model is not None
