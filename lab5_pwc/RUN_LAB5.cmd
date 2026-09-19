@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m unittest test_lab -v
if errorlevel 1 goto failed
".venv\Scripts\python.exe" bootstrap.py
if errorlevel 2 (
  echo Paste the API key into LLM_AUTH_TOKEN, save, then close Notepad.
  start "" /wait notepad.exe ".env"
  pause
) else if errorlevel 1 goto failed
".venv\Scripts\python.exe" bootstrap.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" run_lab.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" verify_results.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" collect_results.py
if errorlevel 1 goto failed
echo Finished. Send lab5_results.zip for review.
pause
exit /b 0
:failed
echo Stopped. Progress is saved. Send the error screenshot WITHOUT the API key.
pause
exit /b 1
