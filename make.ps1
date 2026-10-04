# Windows equivalent of the Makefile. Same target names, same commands.
#
# `make` is not present on a default Windows install and this project is
# developed on Windows, so shipping only a Makefile would mean the documented
# workflow does not run on the machine it is written on. CI (ubuntu-latest) uses
# the Makefile; this is the local counterpart. Keep the two in step.
#
#   .\make.ps1 lint
#   .\make.ps1 test-ci

param([Parameter(Position = 0)][string]$Target = "help")

# Pick an interpreter that actually RUNS, not merely one that exists.
#
# A venv's python.exe is a launcher: it reads pyvenv.cfg and execs the base
# install. Remove or upgrade that base and the file is still on disk but every
# invocation dies with `No Python at '...\python.exe'` -- and the old
# `Test-Path`-only check selected it anyway, so every target failed with an
# error that names a path the user never typed. Probe each candidate, and say
# which one was skipped and why.
function Resolve-Python {
    foreach ($candidate in @(".\.venv\Scripts\python.exe", ".\venv\Scripts\python.exe")) {
        if (-not (Test-Path $candidate)) { continue }
        & $candidate --version 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { return $candidate }
        Write-Host "skipping $candidate -- it exists but cannot run (its base Python was moved or uninstalled; delete the folder and recreate it)" -ForegroundColor Yellow
    }
    return "python"
}

$PY = Resolve-Python
$Trees = @("backend", "dashboard", "train", "data", "evaluation", "tests", "tools")

function Invoke-Step($Description, [scriptblock]$Body) {
    Write-Host "==> $Description" -ForegroundColor Cyan
    & $Body
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Description" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

switch ($Target) {
    "setup" {
        Invoke-Step "install runtime deps" { & $PY -m pip install -r backend/requirements.txt }
        Invoke-Step "install dev deps" { & $PY -m pip install -r backend/requirements-dev.txt }
    }
    "lint" {
        Invoke-Step "ruff" { & $PY -m ruff check @Trees }
    }
    "test" {
        Invoke-Step "pytest" { & $PY -m pytest tests/ -q }
    }
    "test-ci" {
        Invoke-Step "rebuild cheap artifacts" { & $PY train/train_risk_model.py }
        Invoke-Step "rebuild readiness baseline" { & $PY train/train_fleet_readiness.py }
        Invoke-Step "pytest" { & $PY -m pytest tests/ -q -rs }
    }
    "train-fast" {
        Invoke-Step "risk model" { & $PY train/train_risk_model.py }
        Invoke-Step "fleet readiness" { & $PY train/train_fleet_readiness.py }
    }
    "train" {
        Invoke-Step "battery" { & $PY train/train_battery_model.py }
        Invoke-Step "risk" { & $PY train/train_risk_model.py }
        Invoke-Step "readiness" { & $PY train/train_fleet_readiness.py }
        Invoke-Step "casting classifier" {
            & $PY train/train_quality_model.py --data-dir data/raw/casting_defect_clean }
        Invoke-Step "NEU-DET classifier" { & $PY train/train_quality_model_neu_det.py }
    }
    "lock" {
        Invoke-Step "build the API image" {
            & docker build -t ev-fleet-intelligence-api:latest . }
        Invoke-Step "freeze the resolved dependency tree" { & $PY tools/write_lock.py }
    }
    "eval" {
        Invoke-Step "risk lead-time analysis" { & $PY train/analyze_risk_lead_time.py }
    }
    "leakage-check" {
        Invoke-Step "casting split leakage" {
            & $PY evaluation/leakage_check.py data/raw/casting_defect_clean --max-exact 0 }
        Invoke-Step "NEU-DET split leakage" {
            & $PY evaluation/leakage_check.py data/raw/neu_det --max-exact 0 }
    }
    "run-api" {
        Push-Location backend
        try { & "..\$PY" -m uvicorn app.main:app --reload --port 8000 } finally { Pop-Location }
    }
    "run-dashboard" {
        & $PY -m streamlit run dashboard/main.py
    }
    default {
        Write-Host "targets:" -ForegroundColor Cyan
        @(
            @("setup", "install runtime + dev dependencies"),
            @("lint", "ruff over every source tree"),
            @("test", "full suite (needs locally trained vision models for 4 tests)"),
            @("test-ci", "exactly what CI runs: rebuild cheap artifacts, then test"),
            @("train-fast", "the two self-contained pipelines (seconds, no downloads)"),
            @("train", "every pipeline including both ResNet50 classifiers (slow)"),
            @("lock", "freeze exact dependency versions from the built API image"),
            @("eval", "regenerate the analyses behind the published numbers"),
            @("leakage-check", "fail if an image split shares a source part across train/test"),
            @("run-api", "start the backend on :8000"),
            @("run-dashboard", "start the dashboard on :8501")
        ) | ForEach-Object { "  {0,-16} {1}" -f $_[0], $_[1] }
    }
}
