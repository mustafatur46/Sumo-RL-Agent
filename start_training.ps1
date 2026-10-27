<#
.SYNOPSIS
    SumoRLAgent – Training starten

.PARAMETER NumEnvs
    Anzahl paralleler SUMO-Instanzen (Default: 4). 1 = ohne Subprozesse.

.PARAMETER Timesteps
    Gesamt-Trainingsschritte (Default: 2_000_000)

.PARAMETER Resume
    Letzten Checkpoint aus logs\checkpoints fortsetzen

.PARAMETER Gui
    SUMO mit GUI (nur mit -NumEnvs 1 sinnvoll, langsam)

.EXAMPLE
    .\start_training.ps1
    .\start_training.ps1 -NumEnvs 2 -Timesteps 500000
    .\start_training.ps1 -Resume
    .\start_training.ps1 -NumEnvs 1 -Gui
#>
param(
    [int]   $NumEnvs   = 4,
    [int]   $Timesteps = 2000000,
    [switch]$Resume,
    [switch]$Gui
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$python      = "$env:USERPROFILE\miniconda3\envs\sumo-rl\python.exe"

if (-not $env:SUMO_HOME) { $env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo" }
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

foreach ($p in @($python, "$env:SUMO_HOME\bin\sumo.exe", "$projectRoot\oneJunction\oneJunctionLearn.sumocfg")) {
    if (-not (Test-Path $p)) { Write-Host "[FEHLER] nicht gefunden: $p" -ForegroundColor Red; exit 1 }
}

$trainArgs = @("$projectRoot\env\training.py", "--num-envs", $NumEnvs, "--timesteps", $Timesteps)
if ($Resume) { $trainArgs += "--resume" }
if ($Gui)    { $trainArgs += "--gui" }

Write-Host "Starte: python $($trainArgs -join ' ')" -ForegroundColor Cyan
& $python @trainArgs
Write-Host ""
Write-Host "Auswerten:   python env\evaluate.py" -ForegroundColor Cyan
Write-Host "TensorBoard: tensorboard --logdir logs\tb_logs" -ForegroundColor Cyan
