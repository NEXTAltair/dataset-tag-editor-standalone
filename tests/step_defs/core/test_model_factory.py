import pytest
import time
from unittest.mock import patch, MagicMock

# テスト対象のクラス・関数をインポート
from image_annotator_lib.core.model_factory import ModelLoad, BaseModelLoader

# psutil もモック化するのでインポートしておく

# --- pytest-bdd のステップ定義があればここに追加 ---
# (現在は空)

# --- ここからユニットテスト ---


@pytest.fixture(autouse=True)
def clear_model_load_state():
    """各テストの前に ModelLoad のクラス変数をクリアするフィクスチャ"""
    # BaseModelLoader のクリア処理は削除
    # ModelLoad 自身のクラス変数をクリア
    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MODEL_LAST_USED.clear()
    ModelLoad._MODEL_SIZES.clear()
    # テスト用に最大キャッシュサイズを設定 (ModelLoad 経由でアクセスする想定だが、直接設定)
    # ModelLoad._MAX_CACHE_SIZE_GB = 10.0 # _clear_cache_if_needed 内で計算されるため不要
    # ModelLoad._CACHE_RATIO = 0.5 # 同上
    yield  # テスト実行


# --- _clear_cache_if_needed のテスト ---

# psutil.virtual_memory のモック用設定 (MagicMock を使用)
mock_memory = MagicMock()
mock_memory.total = 20 * 1024**3  # 20GB
mock_memory.available = 15 * 1024**3  # 15GB
# 他の属性も必要なら追加 (percent, used, free など)


# patch デコレータの return_value に mock_memory を指定
@patch("image_annotator_lib.core.model_factory.psutil.virtual_memory", return_value=mock_memory)
def test_clear_cache_if_needed_no_release(mock_psutil_vm):
    """キャッシュに十分空きがあり、解放が不要な場合のテスト"""
    loader = BaseModelLoader("new_model", "cpu")
    loader._MEMORY_USAGE["model_a"] = 2 * 1024  # 2GB
    loader._MEMORY_USAGE["model_b"] = 3 * 1024  # 3GB
    # 上限は mock_memory.total * 0.5 = 10GB

    new_model_size_mb = 4 * 1024  # 4GB

    loader._clear_cache_if_needed(new_model_size_mb)

    assert "model_a" in loader._MEMORY_USAGE
    assert "model_b" in loader._MEMORY_USAGE
    assert len(loader._MEMORY_USAGE) == 2


# 同様に他のテストケースも return_value=mock_memory を使う
@patch("image_annotator_lib.core.model_factory.psutil.virtual_memory", return_value=mock_memory)
def test_clear_cache_if_needed_release_oldest(mock_psutil_vm):
    """キャッシュが不足し、最も古いモデルが解放されるテスト"""
    loader = BaseModelLoader("new_model", "cpu")
    # 事前準備
    loader._MEMORY_USAGE["model_a"] = 6 * 1024  # 6GB
    loader._MEMORY_USAGE["model_b"] = 3 * 1024  # 3GB
    loader._MODEL_LAST_USED["model_a"] = time.time() - 100
    loader._MODEL_LAST_USED["model_b"] = time.time()
    # _MODEL_STATES も整合性を取るために設定 (release_model が参照する可能性)
    loader._MODEL_STATES["model_a"] = "on_cpu"
    loader._MODEL_STATES["model_b"] = "on_cpu"

    new_model_size_mb = 2 * 1024  # 2GB -> 合計 11GB > 10GB

    # --- release_model のモック方法を変更 ---
    # モックオブジェクトを作成
    mock_release = MagicMock()

    # side_effect で、呼ばれたモデル名を _MEMORY_USAGE から削除する関数を定義
    def release_side_effect(model_name_to_release):
        if model_name_to_release in loader._MEMORY_USAGE:
            del loader._MEMORY_USAGE[model_name_to_release]
        # release_model は状態 (_MODEL_STATES, _MODEL_LAST_USED) も削除するので、それも模倣
        if model_name_to_release in loader._MODEL_STATES:
            del loader._MODEL_STATES[model_name_to_release]
        if model_name_to_release in loader._MODEL_LAST_USED:
            del loader._MODEL_LAST_USED[model_name_to_release]

    mock_release.side_effect = release_side_effect

    # patch.object で loader.release_model をこのモックに差し替える
    with patch.object(loader, "release_model", mock_release):
        loader._clear_cache_if_needed(new_model_size_mb)

    # 検証: release_model が "model_a" で 1 回だけ呼ばれたか
    mock_release.assert_called_once_with("model_a")
    # 検証: 実際に状態が変わったか確認
    assert "model_a" not in loader._MEMORY_USAGE
    assert "model_a" not in loader._MODEL_STATES
    assert "model_a" not in loader._MODEL_LAST_USED
    assert "model_b" in loader._MEMORY_USAGE  # model_b は残る


# --- 今後、他のケースのテストを追加 ---
