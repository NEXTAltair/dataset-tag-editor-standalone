@echo off
setlocal enabledelayedexpansion

set "VENV_DIR=%~dp0%venv"

:: GPUの確認
nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "HAS_GPU=1"
) else (
    set "HAS_GPU=0"
)

:: バージョン選択メニューの表示
echo PyTorchバージョンを選択してください:
if %HAS_GPU% EQU 1 (
    echo [1] CUDA 11.8
    echo [2] CUDA 12.1
    echo [3] CUDA 12.4
    echo [4] CPU Only
    echo [5] Development mode ^(開発ツール + PyTorch^)
    set /p CHOICE="選択してください (1-5): "
) else (
    echo GPUが検出されませんでした。
    echo [4] CPU Only
    echo [5] Development mode ^(開発ツール + PyTorch^)
    set /p CHOICE="選択してください (4-5): "
)

:process_choice
if "%CHOICE%"=="1" (
    if %HAS_GPU% EQU 0 (
        echo GPUが検出されないため、このオプションは選択できません。
        goto :eof
    )
    set "TORCH_VERSION=cu118"
    set "PYTORCH_INDEX=https://download.pytorch.org/whl/cu118"
) else if "%CHOICE%"=="2" (
    if %HAS_GPU% EQU 0 (
        echo GPUが検出されないため、このオプションは選択できません。
        goto :eof
    )
    set "TORCH_VERSION=cu121"
    set "PYTORCH_INDEX=https://download.pytorch.org/whl/cu121"
) else if "%CHOICE%"=="3" (
    if %HAS_GPU% EQU 0 (
        echo GPUが検出されないため、このオプションは選択できません。
        goto :eof
    )
    set "TORCH_VERSION=cu124"
    set "PYTORCH_INDEX=https://download.pytorch.org/whl/cu124"
) else if "%CHOICE%"=="4" (
    set "TORCH_VERSION=cpu"
    set "PYTORCH_INDEX="
) else if "%CHOICE%"=="5" (
    set "TORCH_VERSION=dev"
    if %HAS_GPU% EQU 1 (
        echo GPUバージョンを選択してください:
        echo [1] CUDA 11.8
        echo [2] CUDA 12.1
        echo [3] CUDA 12.4
        echo [4] CPU Only
        set /p GPU_CHOICE="選択してください (1-4): "

        if "!GPU_CHOICE!"=="1" (
            set "PYTORCH_INDEX=https://download.pytorch.org/whl/cu118"
        ) else if "!GPU_CHOICE!"=="2" (
            set "PYTORCH_INDEX=https://download.pytorch.org/whl/cu121"
        ) else if "!GPU_CHOICE!"=="3" (
            set "PYTORCH_INDEX=https://download.pytorch.org/whl/cu124"
        ) else if "!GPU_CHOICE!"=="4" (
            set "PYTORCH_INDEX="
        ) else (
            echo 無効な選択です。
            goto :eof
        )
    ) else (
        set "PYTORCH_INDEX="
    )
) else (
    echo 無効な選択です。
    goto :eof
)

:: 仮想環境の確認と作成
dir "%VENV_DIR%\Scripts\Python.exe" >nul 2>&1
if %ERRORLEVEL% EQU 0 goto :activate

python -m venv --system-site-packages venv

:activate
call "%VENV_DIR%\Scripts\activate.bat"

:: pipを最新版にアップデート
python -m pip install --upgrade pip

:: 選択されたバージョンでインストール
if "%TORCH_VERSION%"=="dev" (
    echo 開発モード：開発ツールをインストールします...
    pip install -e .[dev]
    echo.
    echo PyTorchをインストールします...
    if defined PYTORCH_INDEX (
        pip install torch -U torchvision torchaudio --index-url !PYTORCH_INDEX!
    ) else (
        pip install torch -U torchvision torchaudio
    )
) else (
    echo %TORCH_VERSION%バージョンをインストールします...
    pip install -e .[%TORCH_VERSION%]
)

echo インストールが完了しました。
pause