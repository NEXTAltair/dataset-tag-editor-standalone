# tests/unit/test_model_factory_unit.py
import pytest
import time
from unittest.mock import patch

# テスト対象のクラス・関数をインポート
from image_annotator_lib.core.model_factory import ModelLoad, BaseModelLoader

# psutil もモック化するのでインポートしておく
import psutil


@pytest.fixture(autouse=True)
def clear_model_load_state():
    """各テストの前に ModelLoad/BaseModelLoader のクラス変数をクリアするフィクスチャ"""
    # BaseModelLoader のクラス変数もクリア対象に追加
    BaseModelLoader._MODEL_STATES.clear()
    BaseModelLoader._MEMORY_USAGE.clear()
    BaseModelLoader._MODEL_LAST_USED.clear()
    BaseModelLoader._MODEL_SIZES.clear()
    # ModelLoad 自身のクラス変数も念のためクリア
    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MODEL_LAST_USED.clear()
    ModelLoad._MODEL_SIZES.clear()
    # テスト用に固定のキャッシュサイズを設定 (BaseModelLoader経由で設定)
    # BaseModelLoader._CACHE_RATIO = 0.5 # これはメソッド内で計算されるので不要かも
    yield  # テスト実行


# --- _clear_cache_if_needed のテスト ---

# psutil.virtual_memory のモック用設定
# total=20GB, available=15GB -> max_cache=10GB (ratio=0.5)
MOCK_VIRTUAL_MEMORY = psutil._common.svmem(
    total=20 * 1024**3,
    available=15 * 1024**3,
    percent=25.0,
    used=5 * 1024**3,
    free=15 * 1024**3,
)


@patch("image_annotator_lib.core.model_factory.psutil.virtual_memory", return_value=MOCK_VIRTUAL_MEMORY)
def test_clear_cache_if_needed_no_release(mock_psutil_vm):
    """キャッシュに十分空きがあり、解放が不要な場合のテスト"""
    # 事前準備: BaseModelLoader のインスタンスを作成し、状態を設定
    loader = BaseModelLoader("new_model", "cpu")
    loader._MEMORY_USAGE["model_a"] = 2 * 1024  # 2GB
    loader._MEMORY_USAGE["model_b"] = 3 * 1024  # 3GB
    # 合計 5GB 使用中、上限は 10GB (MOCK_VIRTUAL_MEMORY.total * 0.5)

    new_model_size_mb = 4 * 1024  # 4GB のモデルをロードしようとする

    # テスト対象メソッド呼び出し
    loader._clear_cache_if_needed(new_model_size_mb)

    # 検証: 何も解放されていないこと
    assert "model_a" in loader._MEMORY_USAGE
    assert "model_b" in loader._MEMORY_USAGE
    assert len(loader._MEMORY_USAGE) == 2  # 変化なし


@patch("image_annotator_lib.core.model_factory.psutil.virtual_memory", return_value=MOCK_VIRTUAL_MEMORY)
def test_clear_cache_if_needed_release_oldest(mock_psutil_vm):
    """キャッシュが不足し、最も古いモデルが解放されるテスト"""
    loader = BaseModelLoader("new_model", "cpu")
    # 事前準備: 状態設定 (model_a が古い)
    loader._MEMORY_USAGE["model_a"] = 6 * 1024  # 6GB
    loader._MEMORY_USAGE["model_b"] = 3 * 1024  # 3GB
    loader._MODEL_LAST_USED["model_a"] = time.time() - 100
    loader._MODEL_LAST_USED["model_b"] = time.time()
    # 合計 9GB 使用中、上限 10GB

    new_model_size_mb = 2 * 1024  # 2GB のモデルをロード -> 合計 11GB で超過

    # release_model が呼ばれることを確認するためにモック化
    with patch.object(loader, "release_model") as mock_release:
        loader._clear_cache_if_needed(new_model_size_mb)

    # 検証: 最も古い model_a が解放されたこと (release_modelが呼ばれたか)
    mock_release.assert_called_once_with("model_a")
    # 検証: (release_modelが正しく動作する前提なら) 状態も確認
    # 注意: release_model をモック化したので、実際のクラス変数は変化しない！
    # 検証したい場合は、release_model をモック化せずに直接呼び出すか、
    # モックの side_effect でクラス変数を操作する必要がある。
    # ここでは release_model が呼ばれたことだけを確認するに留める。


# --- 今後、他のケースのテストを追加 ---
# 例: 解放しても足りない場合、自分自身は解放しない場合など
