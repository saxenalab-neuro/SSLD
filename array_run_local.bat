@echo off
set PYTHONUTF8=1
if not exist ".\result" mkdir ".\result"

for /l %%f in (0,1,4) do (
    python array_area2.py --config array_config.yaml --fold %%f
)
