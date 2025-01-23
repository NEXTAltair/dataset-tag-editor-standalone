@echo off
setlocal enabledelayedexpansion

set "VENV_DIR=%~dp0%venv"

:: GPUとCUDAバージョンの確認
nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "HAS_GPU=1"
    for /f "tokens=9 delims=,. " %%i in ('nvidia-smi --query-gpu^=compute_cap --format^=csv,noheader') do (
        set "GPU_COMPUTE=%%i"
    )
    :: Compute Capabilityに基づいてCUDAバージョンを推奨
    if !GPU_COMPUTE! GEQ 89 (
        set "RECOMMENDED_CUDA=12.4"
    ) else if !GPU_COMPUTE! GEQ 86 (
        set "RECOMMENDED_CUDA=12.1"
    ) else (
        set "RECOMMENDED_CUDA=11.8"
    )
) else (
    set "HAS_GPU=0"
    set "GPU_COMPUTE=なし"
    set "RECOMMENDED_CUDA=なし"
)
echo 検出されたGPU Compute Capability: %GPU_COMPUTE%
echo 推奨CUDAバージョン: %RECOMMENDED_CUDA%
echo.

:: モード選択メニューの表示
echo インストールモードを選択してください:
echo [1] 通常モード
echo [2] 開発モード ^(開発ツール + PyTorch^)
set /p MODE_CHOICE="選択してください (1-2): "

if "%MODE_CHOICE%"=="1" (
    set "IS_DEV_MODE=0"
) else if "%MODE_CHOICE%"=="2" (
    set "IS_DEV_MODE=1"
) else (
    echo 無効な選択です。
    goto :eof
)

:: CUDAバージョン選択メニューの表示
if %HAS_GPU% EQU 1 (
    echo.
    echo あなたのGPU情報:
    echo - Compute Capability: %GPU_COMPUTE%
    echo - 推奨CUDAバージョン: %RECOMMENDED_CUDA%
    echo.
    echo インストールするPyTorchバージョンを選択してください:
    echo [1] CUDA 11.8 ^(Compute Capability 8.0-8.6: RTX 20xx, 30xx前期^)
    echo [2] CUDA 12.1 ^(Compute Capability 8.6-8.9: RTX 30xx後期, 40xx^)
    echo [3] CUDA 12.4 ^(Compute Capability 8.9以上: 最新GPU^)
    echo [4] CPU Only ^(GPU未使用^)
    echo.
    echo 注意: 通常は推奨CUDAバージョンに合わせて選択してください
    set /p CUDA_CHOICE="選択してください (1-4): "
) else (
    echo.
    echo GPUが検出されませんでした。CPU版をインストールします。
    set "CUDA_CHOICE=4"
)

:process_choice
if "%CUDA_CHOICE%"=="1" (
    if %HAS_GPU% EQU 0 (
        echo GPUが検出されないため、このオプションは選択できません。
        goto :eof
    )
    set "TORCH_VERSION=cu118"
    set "PYTORCH_INDEX=https://download.pytorch.org/whl/cu118"
) else if "%CUDA_CHOICE%"=="2" (
    if %HAS_GPU% EQU 0 (
        echo GPUが検出されないため、このオプションは選択できません。
        goto :eof
    )
    set "TORCH_VERSION=cu121"
    set "PYTORCH_INDEX=https://download.pytorch.org/whl/cu121"
) else if "%CUDA_CHOICE%"=="3" (
    if %HAS_GPU% EQU 0 (
        echo GPUが検出されないため、このオプションは選択できません。
        goto :eof
    )
    set "TORCH_VERSION=cu124"
    set "PYTORCH_INDEX=https://download.pytorch.org/whl/cu124"
) else if "%CUDA_CHOICE%"=="4" (
    set "TORCH_VERSION=cpu"
    set "PYTORCH_INDEX="
) else (
    echo 無効な選択です。
    goto :eof
)

if "%IS_DEV_MODE%"=="1" (
    set "TORCH_VERSION=dev"
)

:: 仮想環境の確認と作成
dir "%VENV_DIR%\Scripts\Python.exe" >nul 2>&1
if %ERRORLEVEL% EQU 0 goto :activate

python -m venv venv

:activate
call "%VENV_DIR%\Scripts\activate.bat"

:: pipを最新版にアップデート
python -m pip install --upgrade pip

:: パッケージのインストール
if "%IS_DEV_MODE%"=="1" (
    echo 開発モード：開発ツールをインストールします...
    pip install -e .[dev]
    echo.
)

echo PyTorchをインストールします...
if defined PYTORCH_INDEX (
    pip install torch torchvision torchaudio --index-url !PYTORCH_INDEX!
    pip install -e .
) else (
    pip install torch torchvision torchaudio
    pip install -e .
)

echo インストールが完了しました。
pause