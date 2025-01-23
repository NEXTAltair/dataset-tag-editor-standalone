import os
import sys
from pathlib import Path

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = Path(__file__).parent.parent.resolve()
scripts_path = project_root / "scripts"

# 必要なパスを追加
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(scripts_path))

def pytest_configure(config):
    """テスト実行前の設定"""
    # 元のコマンドライン引数を保存
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]  # スクリプト名のみ残す
sys.path.append(str(project_root / "userscripts"))  # userscriptsディレクトリを後ろに追加