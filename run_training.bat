@echo off
set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
set PATH=%SUMO_HOME%\bin;%PATH%
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d "C:\Users\Mustafa Turhal\OneDrive\Dokumente\SumoRLAgent-main\SumoRLAgent-main"
"C:\Users\Mustafa Turhal\miniconda3\envs\sumo-rl\python.exe" env\training.py --num-envs 1 --timesteps 5000000 > "C:\Users\Mustafa Turhal\OneDrive\Dokumente\SumoRLAgent-main\SumoRLAgent-main\logs\training_live.log" 2>&1
echo Training finished with exit code %errorlevel%
pause
