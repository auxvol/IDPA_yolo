@echo off
chcp 65001 >nul
echo ============================================
echo   IDPA_yolo 環境安裝腳本
echo   Python 3.12 + CUDA 12.8
echo ============================================
echo.

REM --- 1. 檢查 uv 是否已安裝 ---
where uv >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [!] 偵測不到 uv，正在自動安裝...
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if %ERRORLEVEL% neq 0 (
        echo [✗] uv 安裝失敗，請手動安裝後重試。
        echo     https://docs.astral.sh/uv/getting-started/installation/
        pause
        exit /b 1
    )
    echo [✓] uv 安裝完成
) else (
    echo [✓] 已偵測到 uv
)
echo.

REM --- 2. 建立虛擬環境 (Python 3.12) ---
echo [*] 建立虛擬環境 (.venv, Python 3.12)...
uv venv --python 3.12
if %ERRORLEVEL% neq 0 (
    echo [✗] 虛擬環境建立失敗
    pause
    exit /b 1
)
echo [✓] 虛擬環境建立完成
echo.

REM --- 3. 安裝所有依賴套件 ---
echo [*] 安裝依賴套件 (含 PyTorch CUDA 12.8)...
echo     這可能需要幾分鐘，請耐心等待...
echo.
uv pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
if %ERRORLEVEL% neq 0 (
    echo [✗] 套件安裝失敗
    pause
    exit /b 1
)
echo.
echo [✓] 所有套件安裝完成
echo.

REM --- 4. 驗證安裝 ---
echo [*] 驗證關鍵套件...
echo.
.venv\Scripts\python.exe -c "import cv2; print(f'  opencv-contrib-python: {cv2.__version__}')"
.venv\Scripts\python.exe -c "import torch; print(f'  torch: {torch.__version__}  CUDA: {torch.cuda.is_available()}')"
.venv\Scripts\python.exe -c "import ultralytics; print(f'  ultralytics: {ultralytics.__version__}')"
.venv\Scripts\python.exe -c "import scipy; print(f'  scipy: {scipy.__version__}')"
.venv\Scripts\python.exe -c "import numpy; print(f'  numpy: {numpy.__version__}')"
echo.

echo ============================================
echo   安裝完成！
echo.
echo   啟動虛擬環境:
echo     .venv\Scripts\activate
echo.
echo   執行主程式:
echo     python src\main.py
echo     python src\main_pose.py
echo ============================================
pause
