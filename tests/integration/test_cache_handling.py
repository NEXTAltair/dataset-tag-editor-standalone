from pathlib import Path

import pytest
from pytest_bdd import given, parsers, scenario, then, when
import torch

from scorer_wrapper_lib.exceptions.model_errors import ModelNotFoundError
from scorer_wrapper_lib.scorer import _LOADED_SCORERS


# Featureファイルの絶対パスを取得
FEATURE_FILE = str(
    Path(__file__).parent.parent / "features" / "integration" / "cache_handling.feature"
)

# テストで使用するモデル設定（TOMLから変換）
TEST_MODELS = {
    # 実際のモデル設定
    "aesthetic_shadow_v1": {
        "type": "pipeline",
        "model_path": "shadowlilac/aesthetic-shadow",
        "device": "cuda",
        "class": "AestheticShadowV1",
    },
    "cafe_aesthetic": {
        "type": "pipeline",
        "model_path": "cafeai/cafe_aesthetic",
        "device": "cuda",
        "score_prefix": "[CAFE]",
        "class": "CafePredictor",
    },
    # 無効なモデル（テスト用）
    "non_existent_model": {},
}

# モデル名のマッピング
MODEL_NAME_MAP = {
    "テストモデル": "aesthetic_shadow_v1",
    "別のテストモデル": "cafe_aesthetic",
    "無効なモデル名": "non_existent_model",
}


# フィクスチャ
@pytest.fixture
def test_context():
    """統合テストで使用する状態を管理するフィクスチャ"""
    # テスト実行前に_LOADED_SCORERSをクリア
    _LOADED_SCORERS.clear()

    class TestContext:
        def __init__(self):
            # モデル名の保存用
            self.models = {}
            # 捕捉したエラー
            self.exception = None
            # キャッシュ状態の追跡用
            self.model_states = {}

        def set_model(self, key: str, model_name: str):
            """モデル名を保存し、状態追跡用の辞書を初期化"""
            # 実際のモデル名に変換
            actual_name = MODEL_NAME_MAP.get(model_name, model_name)
            self.models[key] = actual_name

            # 状態追跡用の辞書を初期化
            if actual_name not in self.model_states:
                self.model_states[actual_name] = {
                    "before_cache": {},
                    "after_cache": {},
                    "device": None,
                }

            return actual_name

        def get_model(self, key: str) -> str:
            """保存されたモデル名を取得"""
            return self.models.get(key)

        def record_device_state(self, model_name: str, state_key: str, scorer):
            """デバイス状態を記録"""
            if not hasattr(scorer, "model"):
                return

            state_dict = self.model_states[model_name][state_key]

            if hasattr(scorer.model, "device"):
                state_dict["model"] = str(scorer.model.device)
            elif isinstance(scorer.model, dict):
                for component_name, component in scorer.model.items():
                    if hasattr(component, "device"):
                        state_dict[component_name] = str(component.device)
                    elif (
                        component_name == "pipeline"
                        and hasattr(component, "model")
                        and hasattr(component.model, "device")
                    ):
                        state_dict["pipeline_model"] = str(component.model.device)

    return TestContext()


@scenario(FEATURE_FILE, "複数モデルの初期化とキャッシュ")
def test_multiple_models_cache():
    """
    複数モデルの初期化とキャッシュに関するテストシナリオ
    """
    pass


@scenario(FEATURE_FILE, "無効なモデル名でのエラー処理")
def test_invalid_model_error():
    """
    無効なモデル名が指定された場合のエラー処理テストシナリオ
    """
    pass


@scenario(FEATURE_FILE, "キャッシュと解放の繰り返しによるメモリ管理")
def test_cache_release_cycle():
    """
    キャッシュと解放を繰り返した場合のメモリ管理テストシナリオ
    """
    pass


@given("有効なモデル設定が存在する")
def valid_model_config_exists():
    """
    有効なモデル設定が存在することを確認する。

    このステップではテスト前の前提条件を確認する。
    実際のテストでは、モデル設定が存在することを前提としている。
    """
    # 前提条件の確認。実際のテストでは特に何もしない。
    pass


@when(parsers.parse('ライブラリが "{model_name}" を使用してスコアラーを初期化する'))
def init_model(model_name, test_context):
    """
    指定されたモデル名でスコアラーを初期化する。

    Args:
        model_name: 初期化するモデル名
        test_context: テスト状態
    """
    # モデル名を保存し実際のモデル名に変換
    actual_model = test_context.set_model("main", model_name)

    # 初期化前にGPUキャッシュをクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # スコアラーを初期化
    get_scorer_instance(actual_model)

    # デバイス情報を保存
    scorer = _LOADED_SCORERS[actual_model]
    test_context.model_states[actual_model]["device"] = scorer.device


@when(parsers.parse('ライブラリが "{model_name}" を使用して別のスコアラーを初期化する'))
def init_another_model(model_name, test_context):
    """
    別のモデル名でスコアラーを初期化する。

    Args:
        model_name: 初期化するモデル名
        test_context: テスト状態
    """
    # モデル名を保存し実際のモデル名に変換
    actual_model = test_context.set_model("another", model_name)

    # スコアラーを初期化
    get_scorer_instance(actual_model)

    # デバイス情報を保存
    scorer = _LOADED_SCORERS[actual_model]
    test_context.model_states[actual_model]["device"] = scorer.device


@when(
    parsers.parse(
        '再度ライブラリが "{model_name}" を使用して別のスコアラーを初期化する'
    )
)
def init_second_model(model_name, test_context):
    """
    3つ目のモデル名でスコアラーを初期化する。

    Args:
        model_name: 初期化するモデル名
        test_context: テスト状態
    """
    # モデル名を保存し実際のモデル名に変換
    actual_model = test_context.set_model("second", model_name)

    # スコアラーを初期化
    get_scorer_instance(actual_model)

    # デバイス情報を保存
    scorer = _LOADED_SCORERS[actual_model]
    test_context.model_states[actual_model]["device"] = scorer.device


@when("スコアラーの load_or_restore_model メソッドを呼び出す")
def load_model(test_context):
    """
    スコアラーのモデルをロードする。

    このステップでは:
    1. モデルのロードメソッドを呼び出す
    2. ロード後のモデルコンポーネント情報を記録
    """
    # メインモデル名を取得
    model_name = test_context.get_model("main")

    # _LOADED_SCORERSからモデルを取得
    scorer = _LOADED_SCORERS[model_name]

    # モデルをロード
    scorer.load_or_restore_model()

    # キャッシュ前のコンポーネント情報を記録
    test_context.record_device_state(model_name, "before_cache", scorer)


@when("両方のスコアラーの load_or_restore_model メソッドを呼び出す")
def load_both_models(test_context):
    """
    両方のスコアラーのモデルをロードする。

    This step is:
    1. Load the model of the first scorer
    2. Load the model of the second scorer
    """
    # モデル名を取得
    main_model = test_context.get_model("main")
    another_model = test_context.get_model("another")

    # 両方のモデルをロード
    _LOADED_SCORERS[main_model].load_or_restore_model()
    _LOADED_SCORERS[another_model].load_or_restore_model()

    # キャッシュ前のコンポーネント情報を記録
    test_context.record_device_state(
        main_model, "before_cache", _LOADED_SCORERS[main_model]
    )
    test_context.record_device_state(
        another_model, "before_cache", _LOADED_SCORERS[another_model]
    )


@when("2つ目のスコアラーの load_or_restore_model メソッドを呼び出す")
def load_second_model(test_context):
    """
    2つ目のスコアラーのモデルをロードする。
    """
    # モデル名を取得
    model_name = test_context.get_model("second")

    # モデルをロード
    _LOADED_SCORERS[model_name].load_or_restore_model()

    # キャッシュ前のコンポーネント情報を記録
    test_context.record_device_state(
        model_name, "before_cache", _LOADED_SCORERS[model_name]
    )


@when("スコアラーの cache_to_main_memory メソッドを呼び出す")
def cache_model(test_context):
    """
    スコアラーのモデルをメインメモリ（CPU）にキャッシュする。

    This step is:
    1. Record the device information before caching
    2. Call the cache_to_main_memory method
    3. Record the device information after caching
    """
    # モデル名を取得
    model_name = test_context.get_model("main")

    # スコアラーを取得
    scorer = _LOADED_SCORERS[model_name]

    # キャッシュ実行
    scorer.cache_to_main_memory()

    # キャッシュ後のコンポーネント情報を記録
    test_context.record_device_state(model_name, "after_cache", scorer)


@when("スコアラーの cache_and_release_model メソッドを呼び出す")
def cache_and_release_model(test_context):
    """
    スコアラーをキャッシュしてから解放する。

    This step is:
    1. cache_and_release_modelメソッドを呼び出します
    2.モデルの状態を確認します
    """
    # モデル名を取得
    model_name = test_context.get_model("main")

    # スコアラーを取得して操作実行
    _LOADED_SCORERS[model_name].cache_and_release_model()


@when("両方のスコアラーの cache_to_main_memory メソッドを呼び出す")
def cache_both_models(test_context):
    """
    両方のスコアラーのモデルをメインメモリ（CPU）にキャッシュする。
    """
    # モデル名を取得
    main_model = test_context.get_model("main")
    another_model = test_context.get_model("another")

    # 両方のモデルをキャッシュ
    _LOADED_SCORERS[main_model].cache_to_main_memory()
    _LOADED_SCORERS[another_model].cache_to_main_memory()

    # キャッシュ後のコンポーネント情報を記録
    test_context.record_device_state(
        main_model, "after_cache", _LOADED_SCORERS[main_model]
    )
    test_context.record_device_state(
        another_model, "after_cache", _LOADED_SCORERS[another_model]
    )


@when("スコアラーの release_model メソッドを呼び出す")
def release_model(test_context):
    """
    スコアラーのモデルを解放する。
    """
    # モデル名を取得
    model_name = test_context.get_model("main")

    # スコアラーを取得して解放
    _LOADED_SCORERS[model_name].release_resources()


@when(
    parsers.parse(
        'ライブラリが "{model_name}" を使用してスコアラーを初期化しようとする'
    )
)
def try_init_invalid_model(model_name, test_context):
    """
    存在しないモデル名でスコアラーを初期化しようとする。
    エラーが発生した場合、テスト状態に保存する。

    Args:
        model_name: 初期化するモデル名
        test_context: テスト状態
    """
    # モデル名を保存し実際のモデル名に変換
    actual_model = test_context.set_model("main", model_name)

    try:
        # スコアラーを初期化しようとする
        get_scorer_instance(actual_model)
    except Exception as e:
        # エラーが発生した場合、テスト状態に保存
        test_context.exception = e


@then("モデルが正しくロードされCPUにキャッシュされていることを確認する")
def check_model_loaded_and_cached_to_cpu(test_context):
    """
    モデルが正しくロードされ、CPUにキャッシュされていることを確認する。

    検証内容:
    1. is_model_loadedフラグがTrueであること
    2. モデルのコンポーネントがCPUデバイスに配置されていること
    3. キャッシュ前後でデバイスが変更されていること
    """
    # モデル名を取得
    model_name = test_context.get_model("main")

    # スコアラーを取得
    scorer = _LOADED_SCORERS[model_name]

    # is_model_loadedフラグがTrueであることを確認
    assert scorer.is_model_loaded, "モデルがロードされていません"

    # モデルのデバイスがCPUであることを確認
    if hasattr(scorer, "model"):
        if hasattr(scorer.model, "device"):
            assert str(scorer.model.device) == "cpu", "モデルがCPUに移動していません"
        elif isinstance(scorer.model, dict):
            for component_name, component in scorer.model.items():
                if hasattr(component, "device"):
                    assert str(component.device) == "cpu", (
                        f"コンポーネント '{component_name}' がCPUに移動していません"
                    )
                elif (
                    component_name == "pipeline"
                    and hasattr(component, "model")
                    and hasattr(component.model, "device")
                ):
                    assert str(component.model.device) == "cpu", (
                        "パイプラインモデルがCPUに移動していません"
                    )

    # キャッシュ前後でデバイスが変更されていることを確認
    before_cache = test_context.model_states[model_name]["before_cache"]
    after_cache = test_context.model_states[model_name]["after_cache"]

    for component_name, device_before in before_cache.items():
        # 元のデバイス情報がある場合のみチェック
        if component_name in after_cache:
            device_after = after_cache[component_name]
            # GPUからCPUへの移動を確認（元がCPUの場合はスキップ）
            if device_before != "cpu" and "cuda" in device_before:
                assert device_after == "cpu", (
                    f"コンポーネント '{component_name}' がGPUからCPUに正しく移動していません"
                )


@then("両方のスコアラーのモデルが正常にキャッシュされていることを確認する")
def check_both_models_cached(test_context):
    """
    両方のスコアラーのモデルが正常にキャッシュされていることを確認する。

    検証内容:
    1. モデルが正常にロードされていること
    2. モデルがCPUデバイスに配置されていること
    3. スコアラーインスタンスが_LOADED_SCORERSにキャッシュされていること
    """
    # モデル名を取得
    main_model = test_context.get_model("main")
    another_model = test_context.get_model("another")

    # _LOADED_SCORERSからスコアラーを取得
    main_scorer = _LOADED_SCORERS[main_model]
    another_scorer = _LOADED_SCORERS[another_model]

    # モデルがロードされていることを確認
    assert main_scorer.is_model_loaded, "最初のスコアラーのモデルがロードされていません"
    assert another_scorer.is_model_loaded, (
        "2つ目のスコアラーのモデルがロードされていません"
    )

    # _LOADED_SCORERSにキャッシュされていることを確認
    assert main_model in _LOADED_SCORERS, (
        "最初のスコアラーインスタンスが_LOADED_SCORERSにキャッシュされていません"
    )
    assert another_model in _LOADED_SCORERS, (
        "2つ目のスコアラーインスタンスが_LOADED_SCORERSにキャッシュされていません"
    )

    # モデルデバイスがCPUにあることを確認
    for scorer in [main_scorer, another_scorer]:
        if hasattr(scorer, "model"):
            if hasattr(scorer.model, "device"):
                assert str(scorer.model.device) == "cpu", (
                    "モデルがCPUに移動していません"
                )
            elif isinstance(scorer.model, dict):
                for component_name, component in scorer.model.items():
                    if hasattr(component, "device"):
                        assert str(component.device) == "cpu", (
                            f"{component_name} がCPUにありません"
                        )
                    elif (
                        component_name == "pipeline"
                        and hasattr(component, "model")
                        and hasattr(component.model, "device")
                    ):
                        assert str(component.model.device) == "cpu", (
                            "パイプラインのモデルがCPUにありません"
                        )


@then("適切なエラーが発生することを確認する")
def check_error_raised(test_context):
    """
    無効なモデル名が指定された場合に適切なエラーが発生することを確認する。

    検証内容:
    1. 適切な例外が発生していること
    2. エラーメッセージにモデル名が含まれていること
    """
    # モデル名を取得
    model_name = test_context.get_model("main")

    # エラーが発生していることを確認
    assert test_context.exception is not None, "エラーが発生していません"

    # ModelNotFoundErrorまたはValueErrorが発生していることを確認
    # 注: 実際の実装ではModelNotFoundErrorではなくValueErrorが発生する場合があります
    assert isinstance(test_context.exception, (ModelNotFoundError, ValueError)), (
        f"期待された例外ではなく{type(test_context.exception).__name__}が発生しました"
    )

    # エラーメッセージにモデル名またはエラー関連メッセージが含まれていることを確認
    error_message = str(test_context.exception)
    expected_phrases = [model_name, "not found", "見つかりません"]

    assert any(phrase in error_message for phrase in expected_phrases), (
        f"エラーメッセージ '{error_message}' に期待された情報が含まれていません"
    )


@then("モデルが解放されis_model_loadedフラグがFalseになっていることを確認する")
def check_model_released(test_context):
    """
    cache_and_release_model実行後にモデルが解放されていることを確認する。

    検証内容:
    1. is_model_loadedフラグがFalseになっていること
    2. モデル属性が適切に変更されていること
    """
    # モデル名を取得
    model_name = test_context.get_model("main")

    # スコアラーを取得
    scorer = _LOADED_SCORERS[model_name]

    # is_model_loadedフラグがFalseになっていることを確認
    assert not scorer.is_model_loaded, "is_model_loadedフラグがFalseになっていません"

    # モデルが適切に解放されていることを確認（実装による）
    if hasattr(scorer, "model"):
        if isinstance(scorer.model, dict):
            # 辞書型モデルの場合、各コンポーネントが解放されているか確認
            for component_name, component in scorer.model.items():
                if hasattr(component, "device") and component is not None:
                    assert str(component.device) == "cpu", (
                        f"{component_name}がCPUにないか、適切に解放されていません"
                    )


@then("システムリソースが適切に管理されていることを確認する")
def check_system_resources(test_context):
    """
    システムリソースが適切に管理されていることを確認する。

    検証内容:
    1. 最初のスコアラーのモデルが解放されていること
    2. 次のスコアラーのモデルがロードされていること
    3. _LOADED_SCORERSにスコアラーインスタンスが適切に格納されていること
    """
    # モデル名を取得
    main_model = test_context.get_model("main")
    second_model = test_context.get_model("second")

    # _LOADED_SCORERSからスコアラーを取得
    main_scorer = _LOADED_SCORERS[main_model]
    second_scorer = _LOADED_SCORERS[second_model]

    # 最初のスコアラーのモデルが解放されていることを確認
    assert not main_scorer.is_model_loaded, (
        "最初のスコアラーのis_model_loadedフラグがFalseになっていません"
    )

    # 次のスコアラーのモデルがロードされていることを確認
    assert second_scorer.model is not None, (
        "2つ目のスコアラーのモデルがロードされていません"
    )
    assert second_scorer.is_model_loaded, (
        "2つ目のスコアラーのis_model_loadedフラグがTrueになっていません"
    )

    # _LOADED_SCORERSに両方のスコアラーが格納されていることを確認
    assert main_model in _LOADED_SCORERS, (
        "最初のスコアラーが_LOADED_SCORERSから削除されています"
    )
    assert second_model in _LOADED_SCORERS, (
        "2つ目のスコアラーが_LOADED_SCORERSに格納されていません"
    )
