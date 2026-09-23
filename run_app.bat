@echo off
setlocal

cd /d "%~dp0"

set "PROJECT_VENV_PY=%~dp0.venv\Scripts\python.exe"
set "LOCAL_PACKAGES=%~dp0.venv\Lib\site-packages"
set "PYTHON_EXE="
set "PYTHON_ARGS="

if exist "%PROJECT_VENV_PY%" set "PYTHON_EXE=%PROJECT_VENV_PY%"

if not defined PYTHON_EXE (
    for /f "delims=" %%P in ('where py 2^>nul') do (
        if not defined PYTHON_EXE (
            set "PYTHON_EXE=%%P"
            set "PYTHON_ARGS=-3.12"
        )
    )
)

if not defined PYTHON_EXE (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
    )
)

if not defined PYTHON_EXE (
    echo Python was not found on this PC.
    echo Install Python 3.12, run "pip install -r requirements.txt", then run this file again.
    pause
    exit /b 1
)

rem A project .venv is optional: without one, packages from "pip install -r requirements.txt" are used.
if exist "%LOCAL_PACKAGES%" set "PYTHONPATH=%LOCAL_PACKAGES%;%PYTHONPATH%"

set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
set "NO_PROXY=localhost,127.0.0.1,::1"
set "no_proxy=localhost,127.0.0.1,::1"
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"

echo Starting Local Text-to-SQL Agents...
echo.
echo App URL: http://127.0.0.1:8501
echo.
echo Keep this window open while using the app.
echo Press Ctrl+C in this window to stop Streamlit.
echo.

start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 5; Start-Process 'http://127.0.0.1:8501'"
"%PYTHON_EXE%" %PYTHON_ARGS% -m streamlit run app.py

pause
