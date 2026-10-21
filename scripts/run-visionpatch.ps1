$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$visionPython = Join-Path $projectRoot ".venv-vision\Scripts\python.exe"
$existingMlPython = Join-Path $projectRoot ".venv-ml\Scripts\python.exe"
if (Test-Path -LiteralPath $visionPython) {
    & $visionPython -m visionpatch
} elseif (Test-Path -LiteralPath $existingMlPython) {
    & $existingMlPython -m visionpatch
} else {
    throw "Create the isolated environment using docs/visionpatch/README.md first."
}
exit $LASTEXITCODE
