@echo off
title TensorBoard - sumo-rl
echo Starte TensorBoard auf http://localhost:6006 ...
"%USERPROFILE%\miniconda3\envs\sumo-rl\python.exe" -m tensorboard.main --logdir "%~dp0logs\tb_logs" --port 6006
pause
