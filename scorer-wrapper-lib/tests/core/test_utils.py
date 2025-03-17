import logging
import os  # Import os
import tempfile
from pathlib import Path
from typing import Any, cast

from _pytest.capture import CaptureFixture
from _pytest.fixtures import FixtureRequest
from pytest import raises
from pytest_bdd import given, scenarios, then, when

from scorer_wrapper_lib.core.utils import (
    load_file,
    load_model_config,
    setup_logger,
)

scenarios("../../features/core/utils.feature")


# Given steps
@given("ローカルに保存されたファイルがある", target_fixture="local_file_exists")
def local_file_exists() -> str:
    # TODO: これはMockする
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        return tmp_file.name


@given("有効なURLがある", target_fixture="valid_url_exists")
def valid_url_exists() -> str:
    # TODO: これはMockする
    return "https://huggingface.co/google/flan-t5-small/raw/main/config.json"


@given("以前にダウンロードしたファイルがキャッシュに存在する", target_fixture="file_exists_in_cache")
def file_exists_in_cache(valid_url_exists: str) -> str:
    # TODO: これはMockする
    return str(load_file(valid_url_exists))


@given("有効なHuggingFaceHubのリポジトリ名がある", target_fixture="valid_hf_repo_name_exists")
def valid_hf_repo_name_exists() -> str:
    # TODO: これはMockする
    return "google/flan-t5-small"


@given("存在しないパスがある", target_fixture="non_existent_path_exists")
def non_existent_path_exists() -> str:
    return "/path/to/non_existent_file.txt"


@given("無効なURLがある", target_fixture="invalid_url_exists")
def invalid_url_exists() -> str:
    return "https://invalid-url"


@given("存在しないHuggingFaceHubのリポジトリ名がある", target_fixture="non_existent_hf_repo_name_exists")
def non_existent_hf_repo_name_exists() -> str:
    return "non_existent_repo/non_existent_model"


@given("無効なパスまたはURLがある", target_fixture="invalid_path_or_url_exists")
def invalid_path_or_url_exists() -> str:
    return "invalid_path_or_url"


@given("アプリケーション名がある", target_fixture="application_name_exists")
def application_name_exists() -> str:
    return "test_app"


@given("モデル設定ファイル(TOML形式)が存在する", target_fixture="model_config_file_exists")
def model_config_file_exists() -> str:

    config_content = """
[aesthetic_shadow_v2]
type = "pipeline"
model_path = "NEXTAltair/cache_aestheic-shadow-v2"
device = "cuda"
class = "AestheticShadowV2"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as tmp_file:
        tmp_file.write(config_content)
        return tmp_file.name


# When steps
@when("そのパスを指定してファイルにアクセスする")
def access_file_with_path(local_file_exists: str, request: FixtureRequest) -> Path:
    path_type = request.node.callspec.params.get("path_type")
    if path_type == "non_existent":
        path_or_url = request.getfixturevalue("non_existent_path_exists")
    elif path_type == "invalid_path_or_url":
        path_or_url = request.getfixturevalue("invalid_path_or_url_exists")
    else:
        path_or_url = local_file_exists
    return Path(load_file(path_or_url))


@when("そのURLを指定してファイルにアクセスする", target_fixture="accessed_file")
def access_file_with_url(valid_url_exists: str, request: FixtureRequest) -> Path:
    # load_file関数をURLを指定して呼び出す
    url_type = request.node.callspec.params.get("url_type")
    if url_type == "invalid":
        path_or_url = request.getfixturevalue("invalid_url_exists")
    elif url_type == "invalid_path_or_url":
        path_or_url = request.getfixturevalue("invalid_path_or_url_exists")
    else:
        path_or_url = valid_url_exists
    return Path(load_file(path_or_url))


@when("そのリポジトリ名を指定してリポジトリにアクセスする", target_fixture="accessed_file")
def access_repo_with_repo_name(valid_hf_repo_name_exists: str, request: FixtureRequest) -> Path:
    # load_file関数をリポジトリ名を指定して呼び出す
    repo_name_type = request.node.callspec.params.get("repo_name_type")
    if repo_name_type == "non_existent":
        path_or_url = request.getfixturevalue("non_existent_hf_repo_name_exists")
    elif repo_name_type == "invalid_path_or_url":
        path_or_url = request.getfixturevalue("invalid_path_or_url_exists")
    else:
        path_or_url = valid_hf_repo_name_exists
    return Path(load_file(path_or_url))


@when("ロガーをセットアップする", target_fixture="setup_logger_result")
def setup_logger_func(application_name_exists: str) -> logging.Logger:
    # setup_logger関数を呼び出す
    return logging.getLogger(setup_logger(application_name_exists).name)


@when("設定ファイルから設定値にアクセスする", target_fixture="model_config")
def access_config_from_config_file(model_config_file_exists: str) -> dict[str, Any]:
    # load_model_config関数を呼び出す
    model_config: dict[str, dict[str, Any]] = cast(
        dict[str, dict[str, Any]], load_model_config()
    )  # Use typing.cast
    return model_config


# Then steps
@then("絶対パスのPathオブジェクトが返される")
def return_absolute_path_object(accessed_file: Path) -> None:
    # 戻り値がPathオブジェクトであり、絶対パスであることをアサートする
    assert isinstance(accessed_file, Path)
    assert accessed_file.is_absolute()


@then("ファイルがダウンロードされる")
def file_is_downloaded(valid_url_exists: str) -> None:
    # ファイルがダウンロードされたことをアサートする
    downloaded_file_path = Path(load_file(valid_url_exists))
    assert downloaded_file_path.is_file()


@then("ダウンロードしたファイルはモジュールのmodelsディレクトリにキャッシュされる")
def downloaded_file_is_cached_in_models_dir(valid_url_exists: str) -> None:
    # ダウンロードしたファイルがmodelsディレクトリにキャッシュされたことをアサートする
    downloaded_file_path = Path(load_file(valid_url_exists))
    assert "models" in str(downloaded_file_path)


@then("ファイル名はURLのパス部分から抽出される")
def filename_is_extracted_from_url_path(valid_url_exists: str) -> None:
    # ファイル名はURLのパス部分から抽出されることをアサートする
    downloaded_file_path = Path(load_file(valid_url_exists))
    expected_filename = Path(os.path.basename(valid_url_exists))
    assert downloaded_file_path.name == expected_filename.name


@then("新たなダウンロードは行われない")
def no_new_download_is_performed(valid_url_exists: str, capsys: CaptureFixture[str]) -> None:
    # 新たなダウンロードは行われないことをアサートするために、ログ出力を確認する
    load_file(valid_url_exists)
    captured = capsys.readouterr()
    assert "Downloading model from" not in captured.out


@then("リポジトリがダウンロードされる")
def repo_is_downloaded(valid_hf_repo_name_exists: str) -> None:
    # リポジトリがダウンロードされることをアサートする
    repo_path = Path(load_file(valid_hf_repo_name_exists))
    assert repo_path.is_dir()


@then("HuggingFaceHubのキャッシュ機能によって管理される")
def managed_by_hf_hub_cache(valid_hf_repo_name_exists: str) -> None:
    # HuggingFaceHubのキャッシュ機能によって管理されることは、ダウンロード先がキャッシュディレクトリ配下にあることで確認
    repo_path = Path(load_file(valid_hf_repo_name_exists))
    assert "huggingface_hub" in str(repo_path)


@then("ファイルアクセスに失敗する")
def file_access_fails(accessed_file: str) -> None:
    # ファイルアクセスが失敗することをアサートする (例外が発生することを確認)
    with raises(RuntimeError):
        load_file(accessed_file)


@then("適切なエラーメッセージが表示される")
def proper_error_message_is_displayed(accessed_file: str) -> None:
    # 適切なエラーメッセージが表示されたことをアサートする (エラーメッセージの内容を確認)
    with raises(RuntimeError, match="ファイル取得に失敗しました"):
        load_file(accessed_file)


@then("ファイルアクセスに失敗する")
def file_access_fails_again(invalid_path_or_url_exists: str) -> None:
    # ファイルアクセスが失敗することをアサートする (例外が発生することを確認)
    with raises(RuntimeError):
        load_file(invalid_path_or_url_exists)


@then("リポジトリへのアクセスに失敗する")
def repo_access_fails(non_existent_hf_repo_name_exists: str) -> None:
    # リポジトリへのアクセスに失敗することをアサートする (例外が発生することを確認)
    with raises(RuntimeError):
        load_file(non_existent_hf_repo_name_exists)


@then("エラーメッセージには元のパスが含まれる")
def error_message_contains_original_path(invalid_path_or_url_exists: str) -> None:
    # エラーメッセージに元のパスが含まれていることをアサートする (エラーメッセージの内容を確認)
    with raises(
        RuntimeError, match=r"'.+' からのファイル取得に失敗しました"
    ) as exc_info:  # Capture exception info
        assert invalid_path_or_url_exists in str(exc_info.value)  # Assert original path in error message


@then("標準出力にログが記録される")
def log_is_recorded_in_stdout(setup_logger_result: logging.Logger, capsys: CaptureFixture[str]) -> None:
    # 標準出力にログが記録されたことをアサートする
    logger_name = setup_logger_result.name
    captured = capsys.readouterr()
    assert logger_name in captured.out


@then("ログファイルにもログが記録される")
def log_is_recorded_in_logfile(setup_logger_result: logging.Logger) -> None:
    # ログファイルにもログが記録されたことをアサートする
    log_file = Path("logs/scorer_wrapper_lib.log")
    assert log_file.is_file()
    with open(log_file, "r") as f:
        log_content = f.read()
    logger_name = setup_logger_result.name
    assert logger_name in log_content


@then("正しいモデル設定辞書が返される")
def correct_model_config_dict_is_returned(model_config: dict[str, Any]) -> None:
    # 正しいモデル設定辞書が返されることをアサートする
    assert isinstance(model_config, dict)
    assert "model1" in model_config
    assert model_config["model1"]["param1"] == "value1"
    assert model_config["model1"]["param2"] == 123


@then("設定値はキャッシュされ再利用される")
def config_value_is_cached_and_reused(model_config_file_exists: str) -> None:
    # 設定値はキャッシュされ再利用されることをアサートするために、2回呼び出して結果が同じであることを確認
    config1 = load_model_config()
    config2 = load_model_config()
    assert config1 == config2
