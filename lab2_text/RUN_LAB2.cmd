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
if not exist ".env" (
  if exist "..\.env" (
    copy "..\.env" ".env" >nul
  ) else if exist "..\lab1_verified\.env" (
    copy "..\lab1_verified\.env" ".env" >nul
  ) else (
    copy ".env.example" ".env" >nul
    echo Paste your API key into LLM_AUTH_TOKEN in Notepad, save and close it.
    start "" /wait notepad.exe ".env"
    pause
  )
)
".venv\Scripts\python.exe" -m unittest test_lab2 -v
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m unittest discover -s recovery -p test_continue.py -v
if errorlevel 1 goto failed
".venv\Scripts\python.exe" recovery\continue_lab2.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" verify_results.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" collect_results.py
if errorlevel 1 goto failed
echo Finished. Send lab2_results.zip for final review.
pause
exit /b 0
:failed
echo Stopped. Progress is saved. Send the error text or a screenshot without the API key.
pause
exit /b 1
