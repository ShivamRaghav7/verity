param(
    [switch]$PreflightOnly,
    [string]$ResumeCheckpoint
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$mlPython = Join-Path $taskRoot '.venv-ml\Scripts\python.exe'
$runCard = Join-Path $taskRoot 'configs/run_cards/pilot-v3/scope.json'
if (-not (Test-Path -LiteralPath $mlPython)) { throw 'Prepare the isolated ML environment using docs/ML_SETUP.md.' }
if ($ResumeCheckpoint -and $PreflightOnly) { throw 'Resume runs training; preflight alone releases no training.' }
$env:HF_HOME = (Resolve-Path -LiteralPath 'runtime/hf-cache').Path
$env:HF_HUB_OFFLINE = '1'
& $mlPython scripts/preflight_training.py $runCard
if ($LASTEXITCODE -ne 0) { throw 'Scope revision preflight failed; no training started.' }
if ($PreflightOnly) { return }
if ($ResumeCheckpoint) {
    $taskPlan = Get-Content -LiteralPath $runCard -Raw | ConvertFrom-Json
    $taskLatestPointer = Join-Path $taskPlan.output_dir 'latest_checkpoint.json'
    if (-not (Test-Path -LiteralPath $taskLatestPointer)) {
        throw 'Resume requires the latest completed checkpoint from this revision output; no training started.'
    }
    $taskLatest = Get-Content -LiteralPath $taskLatestPointer -Raw | ConvertFrom-Json
    $taskRequested = (Resolve-Path -LiteralPath $ResumeCheckpoint).Path
    $taskLatestPath = (Resolve-Path -LiteralPath $taskLatest.path).Path
    if ($taskRequested -ne $taskLatestPath) {
        throw 'Resume only the latest completed checkpoint. Older states can collide with retained later checkpoints; no training started.'
    }
}
Write-Host 'Starting scope revision 1 of at most 2. Fixed cap: 360 steps. Automated development review; no human-review claim.'
if ($ResumeCheckpoint) {
    & $mlPython scripts/train_auxiliary.py --run-card $runCard --resume $ResumeCheckpoint
} else {
    & $mlPython scripts/train_auxiliary.py --run-card $runCard
}
if ($LASTEXITCODE -ne 0) { throw 'Preserve artifacts and resume from a completed checkpoint using this unchanged card.' }
