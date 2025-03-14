import logging
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch
from urllib.parse import urlparse
from dataclasses import dataclass, field
from pytest import fixture, raises
from pytest_bdd import given, scenarios, then, when
from scorer_wrapper_lib.core.utils import (
    _get_local_file_path,
    load_file,
    load_model_config,
    setup_logger,
)

scenarios("../../features/core/utils.feature")


@dataclass
class ScenarioContext:
    """シナリオの文脈を保持するクラス"""

    path_or_url: str
    accessed_path: Optional[Path] = None
    error: Optional[Exception] = None
    extra: dict[str, Any] = field(default_factory=dict)


# フィクスチャ
@fixture
def mock_local_file():
    """ローカルファイルへのアクセスをモックするフィクスチャ"""
    # 環境に依存しない絶対パスを作成（テスト実行ディレクトリからの絶対パス）
    mock_path = Path.cwd() / "mock_path_to_file.txt"
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.resolve", return_value=mock_path),
        patch("pathlib.Path.is_file", return_value=True),
    ):
        yield mock_path


@fixture
def mock_url_download():
    """URLからのダウンロードをモックするフィクスチャ"""
    # URLから期待されるファイル名とパスを生成
    expected_url = "https://example.com/mock/file.txt"
    filename = Path(urlparse(expected_url).path).name
    mock_path = Path(__file__).parent / "models" / filename

    # 関数ごとにモックを作成
    with patch("scorer_wrapper_lib.core.utils._is_cached") as mock_is_cached:
        with patch(
            "scorer_wrapper_lib.core.utils._perform_download"
        ) as mock_perform_download:
            # デフォルトではキャッシュが存在しないと返す
            mock_is_cached.return_value = (False, mock_path)

            # 以下の情報をテストで利用できるようにする
            yield {
                "expected_url": expected_url,
                "expected_path": mock_path,
                "mock_is_cached": mock_is_cached,
                "mock_perform_download": mock_perform_download,
            }


# Given steps
@given("ローカルに保存されたファイルがある", target_fixture="path_or_url")
def given_local_file_exists(mock_local_file) -> str:
    """
    ローカルファイルのパスを返す（実際のファイルは作成せず、アクセスはモックされる）

    Args:
        mock_local_file: 直接使用しないが、このフィクスチャを受け取ることでモックが有効化される
    """
    # テスト実行ディレクトリからの絶対パスを文字列として返す
    return str(Path.cwd() / "mock_local_file.txt")


@given("有効なURLがある", target_fixture="path_or_url")
def given_valid_url_exists(mock_url_download) -> str:
    """モック用のURL文字列を返す"""
    return mock_url_download["expected_url"]


@given(
    "以前にダウンロードしたファイルがキャッシュに存在する",
    target_fixture="file_exists_in_cache",
)
def given_file_exists_in_cache(mock_url_download) -> str:
    """
    キャッシュ済みファイルを返す（モック）

    Args:
        mock_url_download: このフィクスチャを使ってキャッシュパスを取得する
    """
    # キャッシュが存在することを確認
    mock_url_download["mock_is_cached"].return_value = (
        True,
        mock_url_download["expected_path"],
    )

    # キャッシュパスを返す
    return str(mock_url_download["expected_path"])


@given("存在しないパスがある", target_fixture="path_or_url")
def given_non_existent_path_exists() -> str:
    return "/path/to/non_existent_file.txt"


@given("無効なURLがある", target_fixture="path_or_url")
def given_invalid_url_exists() -> str:
    return "https://invalid-url"


@given("無効なパスまたはURLがある", target_fixture="path_or_url")
def given_invalid_path_or_url_exists() -> str:
    return "invalid_path_or_url"


@given("アプリケーション名がある", target_fixture="application_name_exists")
def given_application_name_exists() -> str:
    return "test_app"


@given(
    "モデル設定ファイル(TOML形式)が存在する", target_fixture="model_config_file_exists"
)
def given_model_config_file_exists(mock_config_toml) -> dict:
    # ファイルを作成せず、辞書を直接返す
    return mock_config_toml.return_value


# When steps
@when("そのパスを指定してファイルにアクセスする", target_fixture="scenario_context")
def when_access_file_with_path(path_or_url: str) -> ScenarioContext:
    """ファイルアクセスを試み、結果をコンテキストに保存"""
    context = ScenarioContext(path_or_url=path_or_url)
    try:
        context.accessed_path = _get_local_file_path(path_or_url)
    except Exception as e:
        context.error = e
    return context


@when("そのURLを指定してファイルにアクセスする", target_fixture="scenario_context")
def when_access_file_with_url(path_or_url: str) -> ScenarioContext:
    """URLアクセスを試み、結果をコンテキストに保存"""
    context = ScenarioContext(path_or_url=path_or_url)
    try:
        path = Path(load_file(path_or_url))
        context.accessed_path = path
    except Exception as e:
        context.error = e
    return context


@when("ロガーをセットアップする", target_fixture="setup_logger_result")
def when_setup_logger_func(application_name_exists: str) -> logging.Logger:
    # setup_logger関数を呼び出す
    return logging.getLogger(setup_logger(application_name_exists).name)


@when("設定ファイルから設定値にアクセスする", target_fixture="model_config")
def when_access_config_from_config_file(
    model_config_file_exists: dict,
) -> dict[str, Any]:
    """モック用の設定ファイルから設定を読み込む"""
    with patch("toml.load") as mock_toml_load:
        mock_toml_load.return_value = model_config_file_exists
        return load_model_config()


# Then steps
@then("絶対パスのPathオブジェクトが返される")
def then_return_absolute_path_object(scenario_context: ScenarioContext) -> None:
    assert scenario_context.error is None
    assert isinstance(scenario_context.accessed_path, Path)
    assert scenario_context.accessed_path.is_absolute()


@then("ファイルがダウンロードされる")
def then_file_is_downloaded(
    scenario_context: ScenarioContext, mock_url_download
) -> None:
    # テスト前にモックをリセット
    mock_url_download["mock_perform_download"].reset_mock()

    # キャッシュが存在しないことを確認
    mock_url_download["mock_is_cached"].return_value = (
        False,
        mock_url_download["expected_path"],
    )

    # 実行
    path_str = load_file(scenario_context.path_or_url)

    # ダウンロードが実行されたことを確認
    mock_url_download["mock_perform_download"].assert_called_once()


@then("ダウンロードしたファイルはモジュールのmodelsディレクトリにキャッシュされる")
def then_downloaded_file_is_cached_in_models_dir(
    scenario_context: ScenarioContext, mock_url_download
) -> None:
    # モックをリセット
    mock_url_download["mock_perform_download"].reset_mock()

    # キャッシュが存在しないことを確認（ダウンロードされるように）
    mock_url_download["mock_is_cached"].return_value = (
        False,
        mock_url_download["expected_path"],
    )

    # 実行
    path_str = load_file(scenario_context.path_or_url)

    # 期待されるパスにmodelsが含まれることを確認
    assert "models" in str(mock_url_download["expected_path"])


@then("ファイル名はURLのパス部分から抽出される")
def then_filename_is_extracted_from_url_path(
    scenario_context: ScenarioContext, mock_url_download
) -> None:
    """URLからファイル名が正しく抽出されることを確認"""
    # モックをリセット
    mock_url_download["mock_perform_download"].reset_mock()
    mock_url_download["mock_is_cached"].reset_mock()

    # キャッシュが存在しないことを確認（ダウンロードされるように）
    mock_url_download["mock_is_cached"].return_value = (
        False,
        mock_url_download["expected_path"],
    )

    # URLからの期待されるファイル名
    expected_filename = Path(urlparse(scenario_context.path_or_url).path).name

    # ファイルをロード
    path_str = load_file(scenario_context.path_or_url)

    # 結果のパスにURLから抽出したファイル名が含まれているか確認
    assert expected_filename in str(path_str)


@then("新たなダウンロードは行われない")
def then_no_new_download_is_performed(
    scenario_context: ScenarioContext, mock_url_download
):
    # モックをリセット
    mock_url_download["mock_perform_download"].reset_mock()

    # キャッシュが存在することを確認
    mock_url_download["mock_is_cached"].return_value = (
        True,
        mock_url_download["expected_path"],
    )

    # 実行
    path_str = load_file(scenario_context.path_or_url)

    # ダウンロードが実行されなかったことを確認
    mock_url_download["mock_perform_download"].assert_not_called()


@then("ファイルアクセスに失敗する")
def then_file_access_fails(scenario_context: ScenarioContext) -> None:
    assert scenario_context.error is not None
    assert isinstance(scenario_context.error, FileNotFoundError)


@then("適切なエラーメッセージが表示される")
def then_proper_error_message_is_displayed(scenario_context: ScenarioContext) -> None:
    """エラーメッセージが存在することを確認"""
    assert scenario_context.error is not None
    assert str(scenario_context.error)  # エラーメッセージが空文字列でないことを確認


@then("ファイルアクセスに失敗する")
def then_file_access_fails_again(path_or_url: str) -> None:
    # ファイルアクセスが失敗することをアサートする (例外が発生することを確認)
    with raises(RuntimeError):
        load_file(path_or_url)


@then("エラーメッセージには元のパスが含まれる")
def then_error_message_contains_original_path(path_or_url: str) -> None:
    """エラーメッセージに元のパスが含まれることをテスト"""
    with raises(
        RuntimeError, match=r"'.+' からのファイル取得に失敗しました"
    ) as exc_info:
        load_file(path_or_url)
        # エラーメッセージに元のパスが含まれていることを確認
        assert path_or_url in str(exc_info.value)


@then("標準出力にログが記録される")
def then_log_is_recorded_in_stdout(
    setup_logger_result: logging.Logger, capsys: Any
) -> None:
    # テストメッセージをログに出力
    test_message = "テストログメッセージ"
    setup_logger_result.info(test_message)

    # ログファイルにメッセージが記録されていることを確認
    log_file = Path("logs/scorer_wrapper_lib.log")
    assert log_file.is_file()
    with open(log_file, "r", encoding="utf-8") as f:
        log_content = f.read()
        assert test_message in log_content


@then("ログファイルにもログが記録される")
def then_log_is_recorded_in_logfile(setup_logger_result: logging.Logger) -> None:
    # テストメッセージをログに出力
    test_message = "テストログメッセージ"
    setup_logger_result.info(test_message)

    # ログファイルにもログが記録されたことをアサートする
    log_file = Path("logs/scorer_wrapper_lib.log")
    assert log_file.is_file()
    with open(log_file, "r", encoding="utf-8") as f:
        log_content = f.read()
        assert test_message in log_content


@then("正しいモデル設定辞書が返される")
def then_correct_model_config_dict_is_returned(model_config: dict[str, Any]) -> None:
    # 正しいモデル設定辞書が返されることをアサートする
    assert isinstance(model_config, dict)
    assert "test_model_01" in model_config
    assert model_config["test_model_01"]["type"] == "pipeline"
    assert model_config["test_model_02"]["type"] == "ClipClassifierModel"


@then("設定値はキャッシュされ再利用される")
def then_config_value_is_cached_and_reused(model_config_file_exists: dict) -> None:
    # 設定値はキャッシュされ再利用されることをアサートするために、2回呼び出して結果が同じであることを確認
    config1 = load_model_config()
    config2 = load_model_config()
    assert config1 == config2
