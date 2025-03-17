"""
スコアラーのBDDテスト用ステップ定義ファイル
"""

from unittest.mock import MagicMock, patch

from behave import given, then, when
from PIL import Image

from scorer_wrapper_lib.scorer import (
    _LOADED_SCORERS,
    _create_scorer_instance,
    _evaluate_model,
    evaluate,
)


@given("有効なクラス名とモデル名が指定される")
def given_valid_class_and_model_names(context):
    # モックの設定
    context.class_name = "TestScorer"
    context.model_name = "test_model"

    # レジストリのモック
    context.mock_registry = {context.model_name: MagicMock()}
    registry_patcher = patch("scorer_wrapper_lib.scorer.get_registry")
    mock_registry = registry_patcher.start()
    mock_registry.return_value = context.mock_registry
    context.add_cleanup(registry_patcher.stop)

    # モデル設定のモック
    context.mock_config = {"parameters": {"param1": "value1"}}
    load_config_patcher = patch("scorer_wrapper_lib.scorer.load_model_config")
    mock_load_config = load_config_patcher.start()
    mock_load_config.return_value = {"models": {context.model_name: context.mock_config}}
    context.add_cleanup(load_config_patcher.stop)


@when("_create_scorer_instance関数を呼び出す")
def when_create_scorer_instance_called(context):
    try:
        context.result = _create_scorer_instance(context.class_name, context.model_name)
        context.exception = None
    except Exception as e:
        context.exception = e
        context.result = None


@then("指定したクラスのスコアラーインスタンスが生成される")
def then_scorer_instance_created(context):
    assert context.result is not None
    assert context.exception is None


@given("無効なクラス名とモデル名が指定される")
def given_invalid_class_and_model_names(context):
    # 無効なクラス名とモデル名
    context.class_name = "NonExistentScorer"
    context.model_name = "test_model"

    # 空のレジストリでモック
    registry_patcher = patch("scorer_wrapper_lib.scorer.get_registry")
    mock_registry = registry_patcher.start()
    mock_registry.return_value = {}
    context.add_cleanup(registry_patcher.stop)


@then("適切なエラーが発生する")
def then_appropriate_error_occurs(context):
    assert context.exception is not None


@given("有効なモデル名が指定される")
def given_valid_model_name(context):
    # 有効なモデル名
    context.model_name = "test_model"

    # 依存関係のモック化
    # _create_scorer_instanceをモック化
    create_scorer_patcher = patch("scorer_wrapper_lib.scorer._create_scorer_instance")
    mock_create_scorer = create_scorer_patcher.start()
    mock_create_scorer.return_value = MagicMock()
    context.add_cleanup(create_scorer_patcher.stop)

    # キャッシュのクリア
    _LOADED_SCORERS.clear()


@when("init_scorer関数を呼び出す")
def when_init_scorer_called(context):
    try:
        context.result = get_scorer_instance(context.model_name)
        context.exception = None
    except Exception as e:
        context.exception = e
        context.result = None


@then("スコアラーが正常に初期化され、キャッシュに保存される")
def then_scorer_initialized_and_cached(context):
    assert context.result is not None
    assert context.model_name in _LOADED_SCORERS


@given("すでに初期化されたモデル名が指定される")
def given_already_initialized_model_name(context):
    # すでに初期化されたモデル名
    context.model_name = "test_model"

    # モックスコアラーをキャッシュに追加
    context.mock_scorer = MagicMock()
    _LOADED_SCORERS[context.model_name] = context.mock_scorer


@then("キャッシュから既存のスコアラーが返される")
def then_existing_scorer_returned_from_cache(context):
    assert context.result is context.mock_scorer


@given("存在しないモデル名が指定される")
def given_nonexistent_model_name(context):
    # 存在しないモデル名
    context.model_name = "nonexistent_model"

    # 例外を発生させるように_create_scorer_instanceをモック化
    create_scorer_patcher = patch("scorer_wrapper_lib.scorer._create_scorer_instance")
    mock_create_scorer = create_scorer_patcher.start()
    mock_create_scorer.side_effect = Exception("Model not found")
    context.add_cleanup(create_scorer_patcher.stop)


@given("有効なスコアラーインスタンスと画像リストが存在する")
def given_valid_scorer_instance_and_image_list(context):
    # モックスコアラーを作成
    context.mock_scorer = MagicMock()
    context.mock_scorer.evaluate.return_value = {"score": 0.8}

    # テスト用の画像リスト（ダミー画像）
    context.images = [MagicMock(spec=Image.Image), MagicMock(spec=Image.Image)]


@when("_evaluate_model関数を呼び出す")
def when_evaluate_model_called(context):
    try:
        context.result = _evaluate_model(context.mock_scorer, context.images)
        context.exception = None
    except Exception as e:
        context.exception = e
        context.result = None


@then("画像ごとの評価結果リストが取得できる")
def then_image_evaluation_results_obtained(context):
    assert context.result is not None
    assert len(context.result) == len(context.images)
    for result in context.result:
        assert "score" in result


@given("画像リストが存在する")
def given_image_list_exists(context):
    # テスト用の画像リスト（ダミー画像）
    context.images = [MagicMock(spec=Image.Image), MagicMock(spec=Image.Image)]


@given("{model_list} のモデルリストが指定される")
def given_model_list_specified(context, model_list):
    # 文字列からリストに変換
    try:
        context.model_list = eval(model_list)
    except:
        context.model_list = []

    # 依存関係のモック化
    # init_scorerをモック化
    init_scorer_patcher = patch("scorer_wrapper_lib.scorer.init_scorer")
    mock_init_scorer = init_scorer_patcher.start()
    mock_scorer = MagicMock()
    mock_scorer.evaluate.return_value = {"score": 0.8}
    mock_init_scorer.return_value = mock_scorer
    context.add_cleanup(init_scorer_patcher.stop)

    # _evaluate_modelをモック化
    evaluate_model_patcher = patch("scorer_wrapper_lib.scorer._evaluate_model")
    mock_evaluate_model = evaluate_model_patcher.start()
    mock_evaluate_model.return_value = [{"score": 0.8}, {"score": 0.9}]
    context.add_cleanup(evaluate_model_patcher.stop)


@when("evaluate関数を呼び出す")
def when_evaluate_called(context):
    try:
        context.result = evaluate(context.images, context.model_list)
        context.exception = None
    except Exception as e:
        context.exception = e
        context.result = None


@then("各モデルの評価結果が辞書形式で取得できる")
def then_evaluation_results_dictionary_obtained(context):
    assert context.result is not None
    assert isinstance(context.result, dict)
    for model_name in context.model_list:
        assert model_name in context.result
        assert len(context.result[model_name]) == len(context.images)


@given("エラーを発生させるモックスコアラーが存在する")
def given_error_causing_mock_scorer(context):
    # エラーを発生させるモックスコアラー
    context.mock_scorer = MagicMock()
    context.mock_scorer.evaluate.side_effect = Exception("Test error")

    # テスト用の画像リスト
    context.images = [MagicMock(spec=Image.Image)]


@then("エラーが適切に処理される")
def then_error_handled_appropriately(context):
    # エラーは吸収されて空の結果が返されるべき
    assert context.result == []


@given("キャッシュされていないモデル名のリストが指定される")
def given_uncached_model_names_list(context):
    # キャッシュされていないモデル名のリスト
    context.model_list = ["uncached_model1", "uncached_model2"]

    # キャッシュのクリア
    _LOADED_SCORERS.clear()

    # init_scorerをモック化
    init_scorer_patcher = patch("scorer_wrapper_lib.scorer.init_scorer")
    mock_init_scorer = init_scorer_patcher.start()
    mock_scorer = MagicMock()
    mock_init_scorer.return_value = mock_scorer
    context.add_cleanup(init_scorer_patcher.stop)

    # _evaluate_modelをモック化
    evaluate_model_patcher = patch("scorer_wrapper_lib.scorer._evaluate_model")
    mock_evaluate_model = evaluate_model_patcher.start()
    mock_evaluate_model.return_value = [{"score": 0.8}]
    context.add_cleanup(evaluate_model_patcher.stop)

    # テスト用の画像リスト
    context.images = [MagicMock(spec=Image.Image)]


@then("自動的にモデルが初期化され評価結果が返される")
def then_models_automatically_initialized(context):
    assert context.result is not None
    for model_name in context.model_list:
        assert model_name in context.result
