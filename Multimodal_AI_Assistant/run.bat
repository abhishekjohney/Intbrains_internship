@echo off
:: Multimodal AI Assistant - Smart Launcher
:: Kills any existing process on port 8000 before starting
chcp 65001 >nul
echo Checking for existing server on port 8000...

for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8000.*LISTENING"') do (
    echo Killing old server process %%a...
    taskkill /PID %%a /F >nul 2>&1
    timeout /t 1 /nobreak >nul
)

echo Starting Multimodal AI Assistant...
echo.
python "%~dp0main.py"
pause
