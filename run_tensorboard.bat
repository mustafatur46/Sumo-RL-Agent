@echo off
title TensorBoard - sumo-rl
call C:\Users\Mustafa Turhal\miniconda3\Scripts\activate.bat sumo-rl
echo Starte TensorBoard...
tensorboard --logdir "C:\Users\Mustafa Turhal\OneDrive\Dokumente\SumoRLAgent-main\SumoRLAgent-main\logs\tb_logs" --port 6006
pause
