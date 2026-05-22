@echo off
set PYTHONUTF8=1
if not exist ".\result" mkdir ".\result"

for /l %%f in (0,1,4) do (
    start "fold_%%f" cmd /k "python array_area2.py --config array_config.yaml --fold %%f"
)
