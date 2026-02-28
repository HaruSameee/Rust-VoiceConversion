@echo off
echo Building RustVC setup.exe...
pip install pyinstaller faiss-cpu tqdm requests --quiet

set INFER_PACK_PATH=infer_pack
if not exist "%INFER_PACK_PATH%" set INFER_PACK_PATH=scripts\infer_pack
set RMVPE_PATH=rmvpe
if not exist "%RMVPE_PATH%" set RMVPE_PATH=scripts\rmvpe

pyinstaller ^
  --onefile ^
  --name setup ^
  --add-data "%INFER_PACK_PATH%;infer_pack" ^
  --add-data "%RMVPE_PATH%;rmvpe" ^
  --add-data "scripts\export_generator_standalone.py;scripts" ^
  --add-data "scripts\convert_index_standalone.py;scripts" ^
  --add-data "scripts\convert_to_ivf.py;scripts" ^
  --hidden-import faiss ^
  --hidden-import tqdm ^
  --hidden-import requests ^
  --exclude-module torch ^
  setup.py

echo.
echo Build complete: dist\setup.exe
pause
