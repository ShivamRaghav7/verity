param(
    [switch]$PreflightOnly,
    [string]$ResumeCheckpoint
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$mlPython = Join-Path $taskRoot '.venv-ml\Scripts\python.exe'
$runCard = Join-Path $taskRoot 'configs/run_cards/targeted-revisions/scope.json'
if (-not (Test-Path -LiteralPath $mlPython)) { throw 'Prepare the isolated ML environment using docs/ML_SETUP.md.' }
if ($ResumeCheckpoint -and $PreflightOnly) { throw 'Resume executes training; preflight alone releases no training.' }
$env:HF_HOME = (Resolve-Path -LiteralPath 'runtime/hf-cache').Path
$env:HF_HUB_OFFLINE = '1'
& $mlPython scripts/preflight_training.py $runCard
if ($LASTEXITCODE -ne 0) { throw 'Final scope revision preflight failed; no training started.' }
if ($PreflightOnly) { return }
if ($ResumeCheckpoint) {
    $taskPlan = Get-Content -LiteralPath $runCard -Raw | ConvertFrom-Json
    $taskLatestPointer = Join-Path $taskPlan.output_dir 'latest_checkpoint.json'
    if (-not (Test-Path -LiteralPath $taskLatestPointer)) { throw 'Resume requires the latest completed checkpoint from this scope revision; no training started.' }
    $taskLatest = Get-Content -LiteralPath $taskLatestPointer -Raw | ConvertFrom-Json
    $taskManifest = Join-Path $taskPlan.output_dir 'manifest.json'
    if ((Test-Path -LiteralPath $taskManifest) -and (Get-Content -LiteralPath $taskManifest -Raw | ConvertFrom-Json).status -eq 'completed') {
        throw 'This scope run is already completed. Preserve its export; no resume or training started.'
    }
    if ($taskLatest.global_step -ge $taskPlan.max_steps) { throw 'The latest checkpoint reached the fixed cap; no resume or training started.' }
    if ((Resolve-Path -LiteralPath $ResumeCheckpoint).Path -ne (Resolve-Path -LiteralPath $taskLatest.path).Path) {
        throw 'Resume only the latest completed checkpoint from this scope revision; no training started.'
    }
}
Write-Host 'Starting scope revision 2 of 2. Fixed cap: 360 steps. Entity-neutral classifier inputs; original approved facts and raw reviewed data retained.'
if ($ResumeCheckpoint) {
    & $mlPython scripts/train_auxiliary.py --run-card $runCard --resume $ResumeCheckpoint
} else {
    & $mlPython scripts/train_auxiliary.py --run-card $runCard
}
if ($LASTEXITCODE -ne 0) { throw 'Preserve artifacts and resume the same card from its latest completed checkpoint.' }
