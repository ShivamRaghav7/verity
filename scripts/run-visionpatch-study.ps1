param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("profile", "develop", "seal", "finalize")]
    [string]$Phase,
    [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$')]
    [string]$RunId = "reproduction-local",
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu"
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$visionPython = Join-Path $projectRoot ".venv-vision\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $visionPython)) {
    $visionPython = Join-Path $projectRoot ".venv-ml\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $visionPython)) {
    throw "Create the isolated environment using README.md first."
}
$runArtifacts = Join-Path $projectRoot "artifacts\visionpatch\local-runs\$RunId"
$runReports = Join-Path $projectRoot "reports\visionpatch\local-runs\$RunId"
Write-Host "Finite VisionPatch study phase: $Phase; run: $RunId; device: $Device"
Write-Host "Run profile first and inspect its measured estimates before develop. Each completed phase refuses overwrite."
& $visionPython -m visionpatch.study $Phase --device $Device --artifacts $runArtifacts --reports $runReports
exit $LASTEXITCODE
