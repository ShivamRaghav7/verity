$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
New-Item -ItemType Directory -Force runtime\tmp | Out-Null
$env:TEMP = (Resolve-Path runtime\tmp).Path
$env:TMP = $env:TEMP
if (-not (Test-Path -LiteralPath .venv\Scripts\python.exe)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the project environment.' }
    .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
.\.venv\Scripts\python.exe -m editguard
