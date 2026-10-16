@echo off
cd /d "%~dp0"
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo [ERROR] Python not found. Please install Python 3 from https://www.python.org/downloads/ and try again.
  pause
  exit /b 1
)
echo Installing edge-tts ...
%PY% -m pip install -q --disable-pip-version-check edge-tts || %PY% -m pip install -q --disable-pip-version-check -i https://pypi.tuna.tsinghua.edu.cn/simple edge-tts
set PYTHONIOENCODING=utf-8
%PY% "%~dp0tts_natural.py"
echo.
pause
