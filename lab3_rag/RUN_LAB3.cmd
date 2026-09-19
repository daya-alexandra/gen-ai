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
  if exist "..\lab2_text\.env" (
    copy "..\lab2_text\.env" ".env" >nul
  ) else if exist "..\.env" (
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
".venv\Scripts\python.exe" -m unittest test_lab3 -v
if errorlevel 1 goto failed
".venv\Scripts\python.exe" pipeline.py all
if errorlevel 1 goto failed
".venv\Scripts\python.exe" verify_results.py --require-api
if errorlevel 1 goto failed
".venv\Scripts\python.exe" collect_results.py
if errorlevel 1 goto failed
echo Finished. Send lab3_results.zip for final review.
pause
exit /b 0
:failed
echo Stopped. Progress is saved. Send the error text or a screenshot without the API key.
pause
exit /b 1
