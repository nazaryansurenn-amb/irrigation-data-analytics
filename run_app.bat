@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE=C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
set "LOCAL_PACKAGES=%~dp0.venv\Lib\site-packages"

if not exist "%PYTHON_EXE%" (
    echo Could not find the local Python runtime:
    echo %PYTHON_EXE%
    echo.
    echo Install Python or update PYTHON_EXE inside run_app.bat.
    pause
    exit /b 1
)

if not exist "%LOCAL_PACKAGES%\streamlit" (
    echo Streamlit packages were not found here:
    echo %LOCAL_PACKAGES%
    echo.
    echo Ask Codex to reinstall the local app dependencies.
    pause
    exit /b 1
)

set "PYTHONPATH=%LOCAL_PACKAGES%"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
set "NO_PROXY=localhost,127.0.0.1,::1"
set "no_proxy=localhost,127.0.0.1,::1"
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"

echo Starting Local AI SQL Intelligence Assistant...
echo.
echo App URL: http://127.0.0.1:8501
echo.
echo Keep this window open while using the app.
echo Press Ctrl+C in this window to stop Streamlit.
echo.

start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 5; Start-Process 'http://127.0.0.1:8501'"
"%PYTHON_EXE%" -m streamlit.web.cli run app.py

pause
