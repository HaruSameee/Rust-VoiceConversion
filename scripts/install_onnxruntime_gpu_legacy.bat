@echo off
setlocal
call "%~dp0install_onnxruntime_provider.bat" cuda11
exit /b %ERRORLEVEL%
