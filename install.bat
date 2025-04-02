@echo off

:: uvがインストールされているか確認
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo uvがインストールされていません。インストールします...
    pip install uv
)

uv venv
uv pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
echo インストールが完了しました。
