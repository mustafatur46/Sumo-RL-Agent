<#
.SYNOPSIS
    Activate the sumo-rl conda environment and set SUMO_HOME for this session.
    Run this at the start of every terminal session before training/evaluation.

.USAGE
    . .\activate.ps1          (dot-source to affect current shell)
#>

# ── Activate conda env ────────────────────────────────────────────────────────
$condaHook = "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
if (Test-Path $condaHook) {
    & $condaHook
    conda activate sumo-rl
} else {
    # Fallback: prepend env\Scripts to PATH directly
    $envBin = "$env:USERPROFILE\miniconda3\envs\sumo-rl\Scripts"
    $env:PATH = "$envBin;$env:PATH"
    Write-Host "[activate] Added $envBin to PATH for this session." -ForegroundColor Yellow
}

# ── SUMO_HOME (adjust path if you installed SUMO elsewhere) ───────────────────
$sumoDefault = "C:\Program Files (x86)\Eclipse\Sumo"
if (-not $env:SUMO_HOME) {
    if (Test-Path "$sumoDefault\bin\sumo.exe") {
        $env:SUMO_HOME = $sumoDefault
        $env:PATH      = "$sumoDefault\bin;$env:PATH"
        Write-Host "[activate] SUMO_HOME = $sumoDefault" -ForegroundColor Green
    } else {
        Write-Host "[activate] SUMO not found at default path." -ForegroundColor Yellow
        Write-Host "           Set manually:  `$env:SUMO_HOME = 'C:\path\to\sumo'" -ForegroundColor Yellow
    }
} else {
    Write-Host "[activate] SUMO_HOME = $env:SUMO_HOME" -ForegroundColor Green
}

Write-Host ""
Write-Host "Environment ready. Start training with:" -ForegroundColor Cyan
Write-Host "  cd env && python training.py --num-envs 2" -ForegroundColor White
Write-Host ""
