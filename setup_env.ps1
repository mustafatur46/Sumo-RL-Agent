<#
.SYNOPSIS
    SumoRLAgent – One-shot environment setup for Windows (PowerShell).

.DESCRIPTION
    1. Locates or creates the conda environment  "sumo-rl"  (Python 3.10).
    2. Installs PyTorch (CUDA 12.4) + all other dependencies.
    3. Sets SUMO_HOME permanently and updates PATH.
    4. Prints a ready-to-use activation command.

.NOTES
    Run ONCE from the project root:
        Set-ExecutionPolicy -Scope CurrentUser RemoteSigned  # (if needed)
        .\setup_env.ps1

    SUMO must already be installed.  Download the Windows installer from:
        https://sumo.dlr.de/docs/Downloads.php#windows
#>

Set-StrictMode -Off
$ErrorActionPreference = "Stop"

# ── 0) Banner ─────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  SumoRLAgent — Environment Setup" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# ── 1) Locate conda ──────────────────────────────────────────────────────────
$condaCandidates = @(
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "C:\ProgramData\miniconda3\Scripts\conda.exe",
    "C:\ProgramData\anaconda3\Scripts\conda.exe"
)
$condaExe = $null
foreach ($c in $condaCandidates) {
    if (Test-Path $c) { $condaExe = $c; break }
}
if (-not $condaExe) {
    Write-Host "[ERROR] conda not found. Install Miniconda3 from:" -ForegroundColor Red
    Write-Host "        https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Yellow
    exit 1
}
Write-Host "[OK] conda found: $condaExe" -ForegroundColor Green
$condaRoot = Split-Path (Split-Path $condaExe)   # …\miniconda3

# ── 2) Create / verify conda env ─────────────────────────────────────────────
$envName = "sumo-rl"
$envPath = "$condaRoot\envs\$envName"
$pip     = "$envPath\Scripts\pip.exe"
$python  = "$envPath\python.exe"

$existing = & $condaExe env list 2>&1 | Select-String $envName
if ($existing) {
    Write-Host "[OK] conda env '$envName' already exists." -ForegroundColor Green
} else {
    Write-Host "[..] Creating conda env '$envName' with Python 3.10 ..." -ForegroundColor Yellow
    & $condaExe create -n $envName python=3.10 -y | Out-Null
    Write-Host "[OK] conda env created." -ForegroundColor Green
}

# ── 3) PyTorch (CUDA 12.4) ───────────────────────────────────────────────────
Write-Host ""
Write-Host "[..] Installing PyTorch 2.6 (CUDA 12.4) ..." -ForegroundColor Yellow
& $pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 `
    --index-url https://download.pytorch.org/whl/cu124 --quiet
Write-Host "[OK] PyTorch installed." -ForegroundColor Green

# ── 4) Remaining Python dependencies ─────────────────────────────────────────
Write-Host "[..] Installing remaining dependencies ..." -ForegroundColor Yellow
$req = Join-Path $PSScriptRoot "requirements.txt"
# Install only the non-commented, non-torch lines
& $pip install `
    "stable-baselines3==2.3.2" `
    "sb3_contrib==2.3.0" `
    "gymnasium==0.29.1" `
    "traci==1.27.0" `
    "sumolib==1.27.0" `
    "numpy==2.2.6" `
    "pandas==2.3.3" `
    "h5py==3.16.0" `
    "matplotlib==3.10.9" `
    "tensorboard==2.20.0" `
    "optuna==4.8.0" `
    "joblib==1.5.3" `
    "psutil==7.2.2" `
    "tqdm==4.67.3" `
    --quiet
Write-Host "[OK] Dependencies installed." -ForegroundColor Green

# ── 5) Locate SUMO and set SUMO_HOME ─────────────────────────────────────────
Write-Host ""
Write-Host "[..] Looking for SUMO installation ..." -ForegroundColor Yellow

$sumoCandidates = @(
    "C:\Program Files (x86)\Eclipse\Sumo",
    "C:\Program Files\Eclipse\Sumo",
    "C:\sumo",
    "$env:USERPROFILE\sumo"
)

# Also search PATH
$sumoInPath = Get-Command sumo.exe -ErrorAction SilentlyContinue
if ($sumoInPath) {
    $fromPath = Split-Path (Split-Path $sumoInPath.Source)
    $sumoCandidates = @($fromPath) + $sumoCandidates
}

$sumoHome = $null
foreach ($s in $sumoCandidates) {
    if (Test-Path "$s\bin\sumo.exe") { $sumoHome = $s; break }
}

if ($sumoHome) {
    Write-Host "[OK] SUMO found: $sumoHome" -ForegroundColor Green

    # Set permanently in user environment
    [System.Environment]::SetEnvironmentVariable("SUMO_HOME", $sumoHome, "User")
    $env:SUMO_HOME = $sumoHome

    # Add to PATH if missing
    $userPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
    $sumobin  = "$sumoHome\bin"
    if ($userPath -notlike "*$sumobin*") {
        [System.Environment]::SetEnvironmentVariable("PATH", "$userPath;$sumobin", "User")
        $env:PATH += ";$sumobin"
        Write-Host "[OK] SUMO\bin added to user PATH." -ForegroundColor Green
    }

    # Also expose SUMO tools to Python (traci fallback)
    $sumoTools = "$sumoHome\tools"
    $userPython = [System.Environment]::GetEnvironmentVariable("PYTHONPATH", "User")
    if ($userPython -notlike "*$sumoTools*") {
        $newPP = if ($userPython) { "$userPython;$sumoTools" } else { $sumoTools }
        [System.Environment]::SetEnvironmentVariable("PYTHONPATH", $newPP, "User")
        Write-Host "[OK] SUMO tools added to PYTHONPATH." -ForegroundColor Green
    }
} else {
    Write-Host ""
    Write-Host "┌─────────────────────────────────────────────────────────┐" -ForegroundColor Yellow
    Write-Host "│  SUMO NOT FOUND – please install it manually:          │" -ForegroundColor Yellow
    Write-Host "│                                                         │" -ForegroundColor Yellow
    Write-Host "│  1. Download: https://sumo.dlr.de/docs/Downloads.php   │" -ForegroundColor Yellow
    Write-Host "│  2. Run the Windows installer (.msi)                   │" -ForegroundColor Yellow
    Write-Host "│  3. Re-run this script – SUMO_HOME will be set auto.   │" -ForegroundColor Yellow
    Write-Host "└─────────────────────────────────────────────────────────┘" -ForegroundColor Yellow
}

# ── 6) Quick sanity check ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "[..] Running import sanity check ..." -ForegroundColor Yellow
$check = & $python -c @"
import sys, os
results = []

def chk(label, fn):
    try:
        fn()
        results.append(f'  OK  {label}')
    except Exception as e:
        results.append(f'  ERR {label}: {e}')

chk('torch + CUDA',        lambda: __import__('torch') and print('       torch', __import__('torch').__version__, '| CUDA', __import__('torch').cuda.is_available()))
chk('stable-baselines3',   lambda: __import__('stable_baselines3'))
chk('sb3_contrib',         lambda: __import__('sb3_contrib'))
chk('gymnasium',           lambda: __import__('gymnasium'))
chk('traci',               lambda: __import__('traci'))
chk('h5py',                lambda: __import__('h5py'))
chk('optuna',              lambda: __import__('optuna'))
chk('psutil',              lambda: __import__('psutil'))
chk('tensorboard',         lambda: __import__('tensorboard'))

for r in results: print(r)

sumo_home = os.environ.get('SUMO_HOME', '')
if sumo_home and os.path.isdir(sumo_home):
    print(f'  OK  SUMO_HOME = {sumo_home}')
else:
    print(f'  --  SUMO_HOME not set or directory missing (install SUMO first)')
"@
Write-Host $check

# ── 7) Done ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Setup complete!  Activate the environment with:" -ForegroundColor Cyan
Write-Host ""
Write-Host "    conda activate sumo-rl" -ForegroundColor White
Write-Host ""
Write-Host "  Then start training:" -ForegroundColor Cyan
Write-Host ""
Write-Host "    cd env" -ForegroundColor White
Write-Host "    python training.py --num-envs 2" -ForegroundColor White
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
