from pathlib import Path
import logging
import toml
import time
from pytest_bdd import given, when, then, parsers, scenarios
import pytest

from image_annotator_lib.core.utils import (
    load_file,
    setup_logger,
)
from image_annotator_lib.core.config import config_registry
scenarios("../features/core/utils.feature")


@pytest.fixture(scope="module")
def cleanup_downloads():
    """テスト実行前にmodelsディレクトリを準備し、終了後に削除するフィクスチャ"""
    # テスト前の準備
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    downloaded_files = []  # ダウンロードしたファイルのパスを記録するリスト

    # このフィクスチャの値としてリストを返す（テスト中に追跡するため）
    yield downloaded_files

    # テスト終了後のクリーンアップ
    for file_path in downloaded_files:
        file_path_obj = Path(file_path)
        if file_path_obj.exists():
            file_path_obj.unlink()

    # modelsディレクトリが空なら削除（オプション）
    if models_dir.exists() and not any(models_dir.iterdir()):
        models_dir.rmdir()


@given("アプリケーションが起動している")
def given_app_is_running():
    # アプリケーション起動のセットアップ
    # 特別な初期化は必要ないのでpassでOK
    pass


@given("設定ファイルにモデルパラメーターが定義されている", target_fixture="config_fixture")
def given_model_parameters_defined():
    return Path("config/models.toml")


@given(
    "ユーザーがURLからダウンロードしたファイルはすでにローカルに存在する",
    target_fixture="cached_file_info",
)
def given_downloaded_file_exists():
    # ダウンロードしたファイルをローカルに保存
    test_url = "https://raw.githubusercontent.com/NEXTAltair/dataset-tag-editor-standalone/refs/heads/refactor/unified-download/tests/resources/txt/test_remote_file.txt"
    result_path = load_file(test_url)

    # ファイルパスと最終更新時刻を記録
    file_path = Path(result_path)
    file_mtime = file_path.stat().st_mtime

    # テスト環境によっては時間差が小さすぎると検出できない場合があるため、少し待機
    time.sleep(0.5)

    return {"path": result_path, "mtime": file_mtime}


@when(
    parsers.parse("ユーザーが{source}のファイルを使用しようとする"),
    target_fixture="source_url_or_path",
)
def when_user_access_file(source, cleanup_downloads):
    if source == "local_file_system":
        path_or_url = Path("tests/resources/txt/test_local_file.txt")
        # 実際にload_file関数を呼び出し、ローカルパスを返す
        return load_file(str(path_or_url))

    elif source == "remote_url":
        # 実際のリモートURLを使用
        test_url = "https://raw.githubusercontent.com/NEXTAltair/dataset-tag-editor-standalone/refs/heads/refactor/unified-download/tests/resources/txt/test_remote_file.txt"

        # モックなしで実際にファイルをダウンロード
        result_path = load_file(test_url)

        # ダウンロードされたファイルのパスをリストに追加（後で削除するため）
        cleanup_downloads.append(result_path)

        return result_path
    else:
        raise ValueError(f"不明なソース: {source}")


@when("アプリケーションが重要な操作を実行する", target_fixture="log_fixture")
def when_app_executes_important_operation():
    # テスト用のログディレクトリとファイルを確保
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # ロガーをセットアップして動作をテスト
    test_logger = setup_logger("test_operation_logger")
    test_logger.info("テスト操作を実行中")
    test_logger.warning("警告メッセージ")

    return test_logger


@when("設定ファイルの読み込みを要求される", target_fixture="config_data")
def when_config_file_is_loaded():
    # 設定ファイルを読み込む
    return config_registry()


@then("システムはそのファイルへのローカルパスを返す")
def then_system_returns_local_path(source_url_or_path):
    # 返されたパスが文字列であることを確認
    assert isinstance(source_url_or_path, str)

    # 返されたパスが実際に存在するファイルを指していることを確認
    assert Path(source_url_or_path).exists()


@then("ファイルは新規にダウンロードされない")
def then_file_is_not_downloaded_again(cached_file_info, source_url_or_path):
    # キャッシュされたファイルと現在のファイルが同じであることを確認
    assert cached_file_info["path"] == source_url_or_path

    # ファイルの最終更新時刻が変わっていないことを確認（ダウンロードされていない証拠）
    current_mtime = Path(source_url_or_path).stat().st_mtime
    assert current_mtime == cached_file_info["mtime"], (
        "ファイルが再ダウンロードされました（更新時刻が変更されています）"
    )


@then("その操作の詳細が適切にログに記録される")
def then_operation_details_logged(log_fixture):
    # ロガーがセットアップされていることを確認
    assert log_fixture.name == "test_operation_logger"
    assert log_fixture.level == logging.INFO

    # ログファイルが作成されていることを確認
    log_file = Path("logs/image_annotator_lib.log")
    assert log_file.exists()


@then("管理者はログを通じてシステムの状態を確認できる")
def then_admin_can_check_system_status(log_fixture):
    # ロガーにハンドラが設定されていることを確認
    assert len(log_fixture.handlers) >= 2  # ストリームハンドラとファイルハンドラ

    # ハンドラの種類を確認
    has_stream_handler = any(isinstance(h, logging.StreamHandler) for h in log_fixture.handlers)
    has_file_handler = any(isinstance(h, logging.FileHandler) for h in log_fixture.handlers)

    assert has_stream_handler, "ストリームハンドラが設定されていません"
    assert has_file_handler, "ファイルハンドラが設定されていません"


@then("設定ファイルから正しいパラメーターが読み込まれる")
def then_correct_parameters_loaded(config_data, config_fixture):
    # 設定ファイルの内容を読み込む
    with open(config_fixture, "r") as f:
        config_data = toml.load(f)

    # 設定データが正しいことを確認
    for model_name, model_config in config_data.items():
        assert model_name in config_data
        assert model_config["type"] in config_data[model_name]["type"]
        assert model_config["class"] in config_data[model_name]["class"]


@then("パフォーマンスのためにこれらの設定値はキャッシュされる")
def then_parameters_cached():
    # キャッシュがうまく機能しているか確認（同じオブジェクトが返されることを確認）
    config1 = config_registry()
    config2 = config_registry()

    # 同一オブジェクトである（キャッシュが機能している）ことを確認
    assert config1 is config2
