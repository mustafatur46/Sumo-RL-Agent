$env:SUMO_HOME = 'C:\Program Files (x86)\Eclipse\Sumo'
$env:PATH = "$env:SUMO_HOME\bin;$env:PATH"
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
Set-Location 'C:\Users\Mustafa Turhal\OneDrive\Dokumente\SumoRLAgent-main\SumoRLAgent-main'
Write-Host 'Starting training...' -ForegroundColor Cyan
& 'C:\Users\Mustafa Turhal\miniconda3\envs\sumo-rl\python.exe' env\training.py --num-envs 1 --timesteps 5000000 *>&1 | Tee-Object -FilePath 'C:\Users\Mustafa Turhal\OneDrive\Dokumente\SumoRLAgent-main\SumoRLAgent-main\logs\training_live.log'
Write-Host 'Training finished.' -ForegroundColor Green
pause
