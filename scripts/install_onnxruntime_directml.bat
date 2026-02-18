@echo off
setlocal
call "%~dp0install_onnxruntime_provider.bat" directml
exit /b %ERRORLEVEL%
