param(
    [ValidateSet('scope', 'answer', 'both')][string]$Task = 'both',
    [switch]$PreflightOnly,
    [string]$ResumeCheckpoint
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$mlPython = Join-Path $taskRoot '.venv-ml\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $mlPython)) { throw 'Prepare the isolated ML environment using docs/ML_SETUP.md.' }
if ($ResumeCheckpoint -and ($Task -eq 'both' -or $PreflightOnly)) {
    throw 'Resume one task at a time with its unchanged card and a completed checkpoint directory.'
}
$env:HF_HOME = (Resolve-Path -LiteralPath 'runtime/hf-cache').Path
$env:HF_HUB_OFFLINE = '1'
$tasksToRun = if ($Task -eq 'both') { @('scope', 'answer') } else { @($Task) }
# Validate all selected cards before the first optimizer run.
foreach ($pilotTask in $tasksToRun) {
    $runCard = Join-Path $taskRoot "configs/run_cards/pilot-v2/$pilotTask.json"
    & $mlPython scripts/preflight_training.py $runCard
    if ($LASTEXITCODE -ne 0) { throw "Preflight failed for $pilotTask; no new training command was released." }
}
if ($PreflightOnly) { return }
foreach ($pilotTask in $tasksToRun) {
    $runCard = Join-Path $taskRoot "configs/run_cards/pilot-v2/$pilotTask.json"
    Write-Host "Starting the finite $pilotTask development run. Review policy: automated_development."
    if ($ResumeCheckpoint) {
        & $mlPython scripts/train_auxiliary.py --run-card $runCard --resume $ResumeCheckpoint
    } else {
        & $mlPython scripts/train_auxiliary.py --run-card $runCard
    }
    if ($LASTEXITCODE -ne 0) { throw "Training stopped for $pilotTask. Preserve artifacts and use the same card for exact resume." }
}
