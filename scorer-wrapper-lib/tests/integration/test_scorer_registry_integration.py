"""スコアラーレジストリモジュールの統合テスト

このモジュールでは、スコアラーレジストリモジュールの統合テストを実装します。
"""

from pathlib import Path

from pytest_bdd import scenario

# Featureファイルの絶対パスを取得
FEATURE_FILE = str(Path(__file__).parent / "features" / "scorer_registry.feature")


# シナリオ定義
@scenario(FEATURE_FILE, "スコアラーの登録")
def test_スコアラーの登録():
    """スコアラー登録のシナリオテスト。"""
    pass


@scenario(FEATURE_FILE, "レジストリの取得")
def test_登録済みスコアラーの取得():
    """登録済みスコアラー取得のシナリオテスト。"""
    pass


@scenario(FEATURE_FILE, "無効なディレクトリからのモジュールファイルリスト取得")
def test_未登録スコアラーの取得():
    """未登録スコアラー取得のシナリオテスト。"""
    pass


@scenario(FEATURE_FILE, "利用可能なスコアラーの一覧取得")
def test_登録済みスコアラーのリスト取得():
    """登録済みスコアラーリスト取得のシナリオテスト。"""
    pass
