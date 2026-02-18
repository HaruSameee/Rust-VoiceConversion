@echo off
setlocal
call "%~dp0install_onnxruntime_provider.bat" cuda
exit /b %ERRORLEVEL%
