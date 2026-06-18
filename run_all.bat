@echo off
REM ====================================================================
REM  One-click runner for Windows -- just double-click this file.
REM  Runs the whole dropout pipeline and writes results/ and figures/.
REM ====================================================================
cd /d "%~dp0"
python run_all.py %*
echo.
echo Done. Press any key to close this window.
pause >nul
