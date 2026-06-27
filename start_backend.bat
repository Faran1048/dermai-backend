@echo off
echo Starting DermAI Backend...
cd /d "%~dp0"
call .venv\Scripts\activate.bat 2>nul || call venv\Scripts\activate.bat 2>nul
.venv\Scripts\uvicorn.exe main:app --reload --host 0.0.0.0 --port 8000
