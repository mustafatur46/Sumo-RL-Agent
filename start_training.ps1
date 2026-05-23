<#
.SYNOPSIS
    SumoRLAgent – Paralleles Training starten (SubprocVecEnv)

.DESCRIPTION
    Setzt SUMO_HOME, fügt das env/-Verzeichnis zum PYTHONPATH hinzu
    und startet training.py mit den gewählten Parametern.

.PARAMETER NumEnvs
    Anzahl paralleler SUMO-Environments (Default: 4)
    - 1  → DummyVecEnv   (kein Overhead, für Debugging)
    - 2  → SubprocVecEnv, ~1.8× schneller
    - 4  → SubprocVecEnv, ~3.2× schneller  ← Empfehlung
    - 8  → SubprocVecEnv, ~5×   schneller  (braucht viel RAM)

.PARAMETER Timesteps
    Gesamt-Trainingsschritte (Default: 5_000_000)

.PARAMETER Gui
    SUMO mit GUI starten (nur für Debugging, stark langsamer!)

.EXAMPLE
    .\start_training.ps1                     # 4 Envs, 5M Schritte
    .\start_training.ps1 -NumEnvs 2          # 2 Envs
    .\start_training.ps1 -NumEnvs 1 -Gui     # Debug mit GUI
    .\start_training.ps1 -Timesteps 10000000 # 10M Schritte
#>

param(
    [int]   $NumEnvs   = 4,
    [int]   $Timesteps = 5_000_000,
    [switch]$Gui
)

Set-StrictMode -Off
$ErrorActionPreference = "Stop"

# ── Banner ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  SumoRLAgent — Paralleles Training" -ForegroundColor Cyan
Write-Host "══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# ── Pfade ─────────────────────────────────────────────────────────────────────
$projectRoot = $PSScriptRoot
$envDir      = Join-Path $projectRoot "env"
$python      = "C:\Users\Mustafa Turhal\miniconda3\envs\sumo-rl\python.exe"
$sumoHome    = "C:\Program Files (x86)\Eclipse\Sumo"
$sumoCfg     = Join-Path $projectRoot "oneJunction\oneJunctionLearn.sumocfg"
$weatherDir  = Join-Path $projectRoot "oneJunction\weatherFiles"
$netFile     = Join-Path $projectRoot "oneJunction\oneJunction.net.xml"

# ── Validierung ───────────────────────────────────────────────────────────────
if (-not (Test-Path $python)) {
    Write-Host "[FEHLER] Python nicht gefunden: $python" -ForegroundColor Red
    Write-Host "         Bitte setup_env.ps1 ausführen." -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path $sumoHome)) {
    Write-Host "[FEHLER] SUMO nicht gefunden: $sumoHome" -ForegroundColor Red
    Write-Host "         SUMO von https://sumo.dlr.de/docs/Downloads.php installieren." -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path $sumoCfg)) {
    Write-Host "[FEHLER] SUMO-Config nicht gefunden: $sumoCfg" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $weatherDir)) {
    Write-Host "[FEHLER] Weather-Verzeichnis nicht gefunden: $weatherDir" -ForegroundColor Red
    exit 1
}

# ── Umgebungsvariablen setzen ─────────────────────────────────────────────────
$env:SUMO_HOME  = $sumoHome
$env:PYTHONPATH = $envDir   # env/ ins Import-Search-Path

Write-Host "[OK] SUMO_HOME   = $sumoHome" -ForegroundColor Green
Write-Host "[OK] PYTHONPATH  = $envDir"   -ForegroundColor Green
Write-Host "[OK] Envs         = $NumEnvs" -ForegroundColor Green
Write-Host "[OK] Timesteps    = $Timesteps" -ForegroundColor Green
if ($Gui) { Write-Host "[OK] GUI          = an (langsam!)" -ForegroundColor Yellow }
Write-Host ""

# Datendurchsatz-Schätzung
$speedup = [math]::Round([math]::Log($NumEnvs + 1) * 2.2, 1)
Write-Host "  ► Erwarteter Speedup vs. 1 Env: ~${speedup}×" -ForegroundColor Cyan
Write-Host "  ► Batch/Rollout: $NumEnvs Envs × 64 Steps = $($NumEnvs * 64) Schritte/Rollout" -ForegroundColor Cyan
Write-Host ""

# ── Trainings-Argumente zusammenstellen ───────────────────────────────────────
$trainArgs = @(
    "training.py",
    "--sumocfg",    $sumoCfg,
    "--net-file",   $netFile,
    "--weather-dir",$weatherDir,
    "--num-envs",   $NumEnvs,
    "--timesteps",  $Timesteps,
    "--log-dir",    (Join-Path $projectRoot "logs"),
    "--tensorboard",(Join-Path $projectRoot "logs\tb_logs")
)
if ($Gui) { $trainArgs += "--gui" }

# ── Training starten ──────────────────────────────────────────────────────────
Write-Host "Starte Training..." -ForegroundColor Yellow
Write-Host "  Befehl: python $($trainArgs -join ' ')" -ForegroundColor DarkGray
Write-Host ""

Set-Location $envDir
try {
    & $python @trainArgs
    Write-Host ""
    Write-Host "[FERTIG] Training abgeschlossen!" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "[FEHLER] Training abgebrochen: $_" -ForegroundColor Red
    exit 1
} finally {
    Set-Location $projectRoot
}

Write-Host ""
Write-Host "  Modell gespeichert in: logs\ppo_sumo_agent.zip" -ForegroundColor Cyan
Write-Host "  TensorBoard starten:   tensorboard --logdir logs\tb_logs" -ForegroundColor Cyan
Write-Host ""
