import io
import logging
from pathlib import Path
import zipfile
import requests
import pytest
from urllib.parse import urlparse
from PIL import Image
import toml
import shutil
from unittest.mock import MagicMock

from pytest_bdd import given, scenarios, then, when, parsers
from image_annotator_lib.core.utils import (
    load_file,
    calculate_phash,
)

scenarios("../../features/core/utils.feature")

# --- 共通の定数とヘルパー ---
error_mapping = {
    "FileNotFound": FileNotFoundError,
    "RequestException": requests.RequestException,
    "Runtime": RuntimeError,
}

# リソースアクセスのステップ定義
RESOURCE_PATHS = {
    "local_image": "test_image.jpg",
    "local_archive": "test_archive.zip",
    "remote_image": "https://example.com/image.jpg",
    "remote_archive": "https://example.com/archive.zip",
    "invalid_path": "invalid/path/file.txt",
    "invalid_url": "https://invalid.url/file.txt",
}

EXPECTED_PATHS = {
    "local_image": "test_image.jpg",
    "local_dir": "test_archive",
    "cache_image": "image.jpg",
    "cache_dir": "archive",
}


def create_test_resource(path: Path, content: str = "test content", is_zip: bool = False):
    """テストリソース作成の共通ヘルパー関数"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_zip:
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("test.txt", content)
        # ZIPの場合は展開後のディレクトリも用意
        extract_dir = path.parent / path.stem
        extract_dir.mkdir(exist_ok=True)
        (extract_dir / "test.txt").write_text(content)
    else:
        path.write_text(content)
    return path


def create_zip_content():
    """ZIPファイルのコンテンツを作成"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("test.txt", "test content")
    return buffer.getvalue()


# --- フィクスチャ ---
@pytest.fixture
def mock_requests(monkeypatch):
    """requests.get を MagicMock でモック化します。"""
    mock_get_instance = MagicMock()

    def mock_get_side_effect(url, stream=False, timeout=None, **kwargs):
        # レスポンス用モックを作成
        response_mock = MagicMock()
        response_mock.url = url
        response_mock.raise_for_status = MagicMock()  # エラーを起こさないようにデフォルト設定

        if "example.com" in url:
            response_mock.status_code = 200
            content = b"test content"
            if url.endswith(".zip"):
                content = create_zip_content()  # ZIPコンテンツ生成ヘルパーを使用
            response_mock.content = content
            response_mock.headers = {"content-length": str(len(content))}
            # iter_content を設定
            response_mock.iter_content = lambda chunk_size=8192: [
                content[i : i + chunk_size] for i in range(0, len(content), chunk_size)
            ]
        elif "invalid.url" in url:
            # ConnectionError を発生させる side_effect を設定
            mock_get_instance.side_effect = requests.exceptions.ConnectionError("Failed to resolve host")
            raise mock_get_instance.side_effect  # 例外を発生させる
        else:
            # 404 エラーを発生させる raise_for_status を設定
            response_mock.status_code = 404
            response_mock.raise_for_status.side_effect = requests.exceptions.HTTPError(
                f"404 Client Error: Not Found for url: {url}"
            )

        return response_mock

    # mock_get_instance の side_effect を設定
    mock_get_instance.side_effect = mock_get_side_effect
    # requests.get をこのモックインスタンスで置き換える
    monkeypatch.setattr(requests, "get", mock_get_instance)
    # モックインスタンス自体を返す (呼び出し履歴などを確認するため)
    yield mock_get_instance


@pytest.fixture
def test_env(tmp_path):
    """テスト環境をセットアップします。"""
    paths = {
        "test_resources": tmp_path / "test_resources",
        "cache": tmp_path / "models",
        "config": tmp_path / "config",
        "logs": tmp_path / "logs",
    }

    for dir_path in paths.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # テスト用の画像を作成
    image_path = paths["test_resources"] / "test_image.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(image_path)

    # テスト用のZIPファイルを作成
    zip_path = paths["test_resources"] / "test_archive.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("test.txt", "test content")

    return {"paths": paths, "files": {"image": image_path, "archive": zip_path}}


@pytest.fixture
def test_image(test_env):
    """テスト用画像の作成"""
    image = Image.new("RGB", (10, 10), color="red")
    image_path = test_env["paths"]["test_resources"] / "test_image.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    return image_path


@pytest.fixture
def test_config(test_env: dict) -> dict:
    """設定ファイルのフィクスチャ"""
    config = {
        "model1": {"param1": "value1", "param2": 100},
        "model2": {"param1": "value2", "param2": 200},
        "test_model": {"param1": "value3", "param2": 300},
    }

    config_file = test_env["paths"]["config"] / "annotator_config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        toml.dump(config, f)

    return {"config": config, "config_file": config_file}


@pytest.fixture
def mock_hf_hub(monkeypatch):
    """Hugging Face Hubのモック"""

    class MockHfHub:
        def __init__(self):
            self.list_repo_files = MagicMock()
            self.hf_hub_download = MagicMock()

    mock = MockHfHub()
    monkeypatch.setattr("huggingface_hub.list_repo_files", mock.list_repo_files)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", mock.hf_hub_download)
    return mock


@pytest.fixture
def mock_model_files(test_env, monkeypatch):
    """モデルファイルのモック"""
    model_dir = test_env["paths"]["cache"]
    model_dir.mkdir(parents=True, exist_ok=True)

    # モデルファイルを作成
    model_files = {
        "model.onnx": b"mock onnx content",
        "config.json": '{"model_type": "test"}',
        "vocab.txt": "test vocabulary",
        "tags.csv": "tag,count\ntest,1",
    }

    for filename, content in model_files.items():
        file_path = model_dir / filename
        if isinstance(content, bytes):
            with open(file_path, "wb") as f:
                f.write(content)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

    return model_dir


# --- ステップ定義 ---
@given("アプリケーションを起動している", target_fixture="app_state")
def given_app_running():
    """アプリケーション起動状態の設定"""
    return {"logger": logging.getLogger("test_logger")}


@given("ログレベルがDEBUGに設定されている", target_fixture="log_config")
def given_debug_level(test_env: dict):
    """ログレベルの設定"""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # ファイルハンドラの設定
    log_file = test_env["paths"]["logs"] / "image-annotator-lib.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return {"logger": logger, "level": "DEBUG"}


@given(parsers.parse("リソース{source_type}が{location}に存在する"), target_fixture="resource_info")
def given_resource_exists(source_type: str, location: str, test_env: dict) -> dict:
    """リソース存在確認"""
    is_remote = source_type.startswith("remote")
    is_zip = source_type.endswith("zip")
    resource_key = location
    path_str_or_url = RESOURCE_PATHS[resource_key]

    resources_dir = test_env["paths"]["test_resources"]
    cache_dir = test_env["paths"]["cache"]
    actual_file_path = None

    if not is_remote:
        # Local resource
        actual_file_path = resources_dir / path_str_or_url
        if resource_key != "invalid_path":
            # ローカルファイルの内容は区別できるようにする
            content = "test content" if is_zip else "dummy content for local file"
            create_test_resource(actual_file_path, content=content, is_zip=is_zip)
        path_to_load = str(actual_file_path)
    else:
        # Remote resource (URL)
        path_to_load = path_str_or_url

    return {
        "path_to_load": path_to_load,
        "actual_file_path": actual_file_path,
        "is_remote": is_remote,
        "is_zip": is_zip,
        "source_type": source_type,
        "resources_dir": resources_dir,
        "cache_dir": cache_dir,
        "test_env": test_env,
        "location_key": resource_key,
    }


@given(parsers.parse("キャッシュの状態が{state}"))
def given_cache_state(state: str, resource_info: dict):
    """キャッシュの状態を設定"""
    cache_dir = resource_info["cache_dir"]  # resource_info から直接取得
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 期待されるキャッシュパスを決定
    path_to_load = resource_info["path_to_load"]
    is_zip = resource_info["is_zip"]
    is_remote = resource_info["is_remote"]

    # リモートの場合のみキャッシュを考慮
    if is_remote:
        path_obj = Path(urlparse(path_to_load).path)  # URLからパス部分を取得
        expected_cache_filename = path_obj.name
        if not expected_cache_filename:  # URL にパスがない場合など
            expected_cache_filename = "archive.zip" if is_zip else "file.bin"

        if is_zip:
            # ZIPの場合、キャッシュパスは展開後のディレクトリ
            expected_cache_path = cache_dir / Path(expected_cache_filename).stem
        else:
            # 通常ファイルの場合、キャッシュパスはファイル名
            expected_cache_path = cache_dir / expected_cache_filename

        # 既存のキャッシュを削除して状態をリセット
        if expected_cache_path.is_file():
            expected_cache_path.unlink()
        elif expected_cache_path.is_dir():
            shutil.rmtree(expected_cache_path)

        resource_info["cache_file"] = None  # デフォルトはキャッシュなし

        if state == "exists":
            logging.debug(f"Setting cache state to 'exists' for: {expected_cache_path}")
            if is_zip:
                # 展開後ディレクトリとダミーファイルを作成
                expected_cache_path.mkdir(exist_ok=True)
                (expected_cache_path / "test.txt").write_text("test content")
                resource_info["cache_file"] = expected_cache_path  # ディレクトリパスを格納
            else:
                # 通常のキャッシュファイルを作成
                expected_cache_path.write_bytes(b"test content")
                resource_info["cache_file"] = expected_cache_path  # ファイルパスを格納
        elif state == "empty":
            logging.debug(f"Ensuring cache state is 'empty' for: {expected_cache_path}")
    else:
        # ローカルリソースの場合はキャッシュ状態を 'none' とし、何もしない
        resource_info["cache_file"] = None
        logging.debug("Local resource, skipping cache state setup.")


@when("システムがリソースパスを解決する", target_fixture="resolved_path")
def when_resolve_path(resource_info: dict, mock_requests: MagicMock) -> Path:
    """パス解決の実行"""
    # 呼び出し情報をリセット
    mock_requests.reset_mock()
    resource_info["mock_get_called"] = False
    resource_info["mock_get_call_args"] = None

    path_to_load = resource_info["path_to_load"]  # 使用するパス/URL
    cache_dir = resource_info["cache_dir"]  # キャッシュディレクトリ

    try:
        # load_file を呼び出す
        resolved = load_file(path_to_load, cache_dir=cache_dir)
        # 呼び出し情報を resource_info に保存
        resource_info["mock_get_called"] = mock_requests.called
        resource_info["mock_get_call_args"] = mock_requests.call_args
        return resolved
    except (RuntimeError, requests.RequestException) as e:
        resource_info["error"] = e
        # エラー時も呼び出し情報を記録
        resource_info["mock_get_called"] = mock_requests.called
        resource_info["mock_get_call_args"] = mock_requests.call_args
        # エラー時のパス返却処理 (エラーハンドリングは then で行うため、ここでは適当な値を返す)
        return Path(f"error_{Path(path_to_load).name}")  # 例: error_image.jpg


@then(parsers.parse("{action}が実行される"))
def then_verify_action(action: str, resource_info: dict, resolved_path: Path):  # resolved_path を引数に追加
    """アクション実行の検証"""
    if action == "download":
        # resource_info から呼び出し情報を取得して確認
        assert resource_info.get(
            "mock_get_called", False
        ), "HTTPリクエスト (requests.get) が呼び出されていません"
        call_args = resource_info.get("mock_get_call_args")
        assert call_args is not None, "HTTPリクエストの呼び出し引数が記録されていません"
        # 呼び出し引数の URL を検証 (resource_info["path_to_load"] と一致するか)
        requested_url = call_args.args[0]
        expected_url = resource_info["path_to_load"]
        assert (
            requested_url == expected_url
        ), f"予期しないURLへのリクエスト: {requested_url} (期待: {expected_url})"
    elif action == "extract_zip":
        # when で返された resolved_path がディレクトリであることを確認
        # resolved_path は引数で受け取る
        assert resolved_path.is_dir(), f"ZIP展開後のパスがディレクトリではありません: {resolved_path}"
        # 展開後のファイル存在確認
        assert (resolved_path / "test.txt").exists(), "展開された test.txt が見つかりません"
    elif action == "read_local":
        # when で返された resolved_path がファイルであることを確認
        # resolved_path は引数で受け取る
        assert resolved_path.is_file(), f"ローカルリソースのパスがファイルではありません: {resolved_path}"
        assert resolved_path.exists()  # exists は when で確認済みだが念のため
    elif action == "use_cache":
        # when で返された resolved_path がキャッシュパスと一致するか確認
        # resolved_path は引数で受け取る
        expected_cache_path = resource_info.get("cache_file")
        assert (
            expected_cache_path is not None
        ), "キャッシュファイルパスが resource_info に設定されていません"
        assert (
            resolved_path.resolve() == expected_cache_path.resolve()
        ), f"解決されたパスがキャッシュパスと一致しません。\n期待: {expected_cache_path}\n実際: {resolved_path}"


@then(parsers.parse("保存されているリソースのパス{expected_path_key}を返す"))
def then_verify_path(expected_path_key: str, resolved_path: Path, resource_info: dict):
    """パスの検証"""
    # EXPECTED_PATHS から期待される相対パス部分を取得
    expected_suffix = EXPECTED_PATHS[expected_path_key]
    cache_dir = resource_info["cache_dir"]
    resources_dir = resource_info["resources_dir"]

    # 期待されるフルパスを計算
    if resource_info["is_remote"]:
        # リモートの場合、キャッシュディレクトリ基準
        expected_full_path = (cache_dir / expected_suffix).resolve()
    else:
        # ローカルの場合、リソースディレクトリ基準
        expected_full_path = (resources_dir / expected_suffix).resolve()

    # 実際の解決済みパスと比較
    assert (
        resolved_path.resolve() == expected_full_path
    ), f"解決されたパスが期待値と異なります。\n期待: {expected_full_path}\n実際: {resolved_path.resolve()}"


# --- エラー処理のテスト ---
# location 引数を追加し、resource_info への依存を減らす
@given(parsers.parse("アクセス状態が{condition}"))
def given_access_condition(
    condition: str, resource_info: dict, test_env: dict
):  # location は Feature から暗黙的に渡される想定だが、明示的に引数に加える方が安全かもしれない。ここでは resource_info から取得する。
    """アクセス状態の設定"""
    location_key = resource_info.get("location_key")  # given_resource_exists で設定される想定のキー
    if not location_key:
        pytest.skip(
            "location_key not found in resource_info for given_access_condition"
        )  # スキップするかエラーにする

    path_str_or_url = RESOURCE_PATHS.get(location_key)
    if not path_str_or_url:
        pytest.fail(f"Invalid location_key '{location_key}' found in resource_info")

    is_remote = resource_info.get("is_remote", False)

    if condition == "missing":
        if not is_remote:
            # ローカルファイルの場合、test_env と path_str_or_url からパスを特定
            path = test_env["paths"]["test_resources"] / path_str_or_url
            if path.exists():
                logging.debug(f"Removing local file/dir for 'missing' state: {path}")
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
        else:
            # リモートファイルの場合、キャッシュを削除
            cache_dir = test_env["paths"]["cache"]  # test_env から取得
            path_obj = Path(urlparse(path_str_or_url).path)
            expected_cache_filename = path_obj.name
            if not expected_cache_filename:
                expected_cache_filename = "archive.zip" if resource_info.get("is_zip") else "file.bin"

            if resource_info.get("is_zip"):
                expected_cache_path = cache_dir / Path(expected_cache_filename).stem
            else:
                expected_cache_path = cache_dir / expected_cache_filename

            if expected_cache_path.exists():
                logging.debug(f"Removing cache file/dir for 'missing' state: {expected_cache_path}")
                if expected_cache_path.is_file():
                    expected_cache_path.unlink()
                elif expected_cache_path.is_dir():
                    shutil.rmtree(expected_cache_path)
            # resource_info["cache_file"] は given_cache_state で None に設定されるはず


@then(parsers.parse("{error_type}エラーが発生する"))
def then_error_occurs(error_type: str, resource_info: dict):
    """エラー発生の検証"""
    assert "error" in resource_info
    error = resource_info["error"]

    if error_type == "file_not_found":
        assert isinstance(error, RuntimeError)
        assert "ローカルファイルが見つかりません" in str(error)
    elif error_type == "request_error":
        assert isinstance(error, RuntimeError)
        assert "URLからのダウンロードに失敗しました" in str(error)
        assert "Failed to resolve host" in str(error)


# --- ユーティリティ機能のテスト ---
@when("アプリケーションが重要な操作を実行する", target_fixture="operation_log")
def when_perform_operation(log_config: dict) -> str:
    """重要な操作の実行"""
    message = "重要な操作が実行されました"
    log_config["logger"].debug(message)
    return message


@then("その操作の詳細が適切にログに記録される")
def then_verify_log(operation_log: str, test_env: dict):
    """ログ記録の検証"""
    log_content = test_env["paths"]["logs"] / "image-annotator-lib.log"
    assert operation_log in log_content.read_text()


@then("ログエントリにはタイムスタンプと操作の種類が含まれる")
def then_verify_log_format(test_env: dict):
    """ログフォーマットの検証"""
    log_content = test_env["paths"]["logs"] / "image-annotator-lib.log"
    assert "DEBUG" in log_content.read_text()
    assert "test_logger" in log_content.read_text()


# --- 設定ファイル管理のテスト ---
@given("設定ファイルにパラメーターが定義されている", target_fixture="config")
def given_config_params(test_config: dict) -> dict:
    """設定ファイルの準備"""
    return test_config


@when("設定ファイルの読み込みを要求される", target_fixture="loaded_config")
def when_load_config(test_config: dict) -> dict:
    """設定ファイルの読み込み"""
    return load_model_config(test_config["config_file"])


@then("設定ファイルから正しいパラメーターが読み込まれる")
def then_verify_config(loaded_config: dict, config: dict):
    """設定パラメーターの検証"""
    assert loaded_config == config["config"]


# --- 画像処理のテスト ---
@given("ユーザーが画像ファイルを使用しようとする", target_fixture="image_path")
def given_image_file(test_image: Path) -> Path:
    """画像ファイルの準備"""
    return test_image


@when("システムが画像のpHashを計算する", target_fixture="calculated_phash")
def when_calculate_phash(image_path: Path) -> str:
    """pHashの計算"""
    with Image.open(image_path) as img:
        return calculate_phash(img)


@then("システムは有効なpHash文字列を返す")
def then_verify_phash(calculated_phash: str):
    """pHash文字列の検証"""
    assert isinstance(calculated_phash, str)
    assert len(calculated_phash) > 0
    assert all(c in "0123456789abcdef" for c in calculated_phash)


# --- Hugging Face関連のテスト ---
@given(parsers.parse("Hugging Face Hubにモデルリポジトリが存在する"), target_fixture="repo_info")
def given_hf_repo(mock_hf_hub):
    """HFリポジトリの準備"""
    return {"repo_id": "test/model-repo"}


@given("そのリポジトリに必要なファイルが含まれる")
def given_repo_files(mock_hf_hub, repo_info):
    """リポジトリファイルの設定"""
    mock_hf_hub.list_repo_files.return_value = ["model.onnx", "tags.csv"]
    mock_hf_hub.hf_hub_download.return_value = str(Path.home() / "models" / repo_info["repo_id"])


@when("システムがONNX Taggerモデルのダウンロードを試みる", target_fixture="model_paths")
def when_download_model(repo_info):
    """モデルのダウンロード"""
    from image_annotator_lib.core.utils import download_onnx_tagger_model

    return download_onnx_tagger_model(repo_info["repo_id"])


@then("モデルファイルとラベルファイルのパスが返される")
def then_verify_model_files(mock_model_files):
    """モデルファイルの検証"""
    assert mock_model_files.exists()
    assert (mock_model_files / "model.onnx").exists()
    assert (mock_model_files / "tags.csv").exists()


@then("Hugging Face Hubのダウンロード関数が適切な引数で呼び出される")
def then_verify_hf_calls(mock_hf_hub, repo_info):
    """HF関数呼び出しの検証"""
    assert mock_hf_hub.list_repo_files.called
    assert mock_hf_hub.hf_hub_download.call_count == 2
    calls = mock_hf_hub.hf_hub_download.call_args_list
    assert all(call.args[0] == repo_info["repo_id"] for call in calls)
    filenames = {call.args[1] for call in calls}
    assert filenames == {"model.onnx", "tags.csv"}


# --- モデルサイズ管理のテスト ---
@given("モデルの設定が存在する")
def given_model_exists(test_config: dict):
    """モデル設定の確認"""
    assert test_config


@given("モデルのサイズが計算されている", target_fixture="model_size")
def given_model_size() -> float:
    """モデルサイズの設定"""
    return 512.0


@when("システムがモデルのサイズを保存する")
def when_save_model_size(test_config: dict):
    """モデルサイズの保存"""
    config = toml.load(test_config["config_file"])
    config["test_model"]["estimated_size_gb"] = 0.512
    with open(test_config["config_file"], "w") as f:
        toml.dump(config, f)


@then("モデルの推定サイズが保存される")
def then_verify_saved_size(test_config: dict):
    """保存されたサイズの検証"""
    updated_config = toml.load(test_config["config_file"])
    assert "estimated_size_gb" in updated_config["test_model"]
    saved_size = updated_config["test_model"]["estimated_size_gb"]
    assert isinstance(saved_size, float)
