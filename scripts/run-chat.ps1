param([switch]$Setup, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$taskProjectPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskProjectPath
$taskPythonPath = Join-Path $taskProjectPath '.venv-chat\Scripts\python.exe'

try {
    $taskOllamaStatus = Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 3
} catch {
    $taskOllamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $taskOllamaCommand) { throw 'Install Ollama from https://ollama.com/download/windows, open it, and run this script again.' }
    Start-Process -FilePath $taskOllamaCommand.Source -ArgumentList 'serve' -WindowStyle Hidden | Out-Null
    $taskOllamaStatus = $null
    for ($taskAttempt=0; $taskAttempt -lt 20; $taskAttempt++) {
        try { $taskOllamaStatus = Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2; break } catch { Start-Sleep -Seconds 1 }
    }
    if (-not $taskOllamaStatus) { throw 'Open Ollama and try again; its local server is not ready on port 11434.' }
}

if ($Setup) {
    if (-not (Test-Path -LiteralPath $taskPythonPath)) {
        $taskExistingPython = Join-Path $taskProjectPath '.venv\Scripts\python.exe'
        if (Test-Path -LiteralPath $taskExistingPython) { & $taskExistingPython -m venv .venv-chat }
        else { python -m venv .venv-chat }
        if ($LASTEXITCODE -ne 0) { throw 'Creating the isolated Python environment failed.' }
    }
    & $taskPythonPath -m pip --isolated install --index-url https://pypi.org/simple -r requirements-chat.txt
    if ($LASTEXITCODE -ne 0) { throw 'Image validation dependency installation failed.' }
    Write-Host 'Preparing official pretrained chat and embedding models. This downloads model files and runs no training.'
    & $taskPythonPath -m scripts.prepare_chat_models --download
    if ($LASTEXITCODE -ne 0) { throw 'Model preparation failed; inspect the message above. Valid cached files are retained.' }
    Write-Host 'Setup complete. Run .\scripts\run-chat.ps1 to open EditGuard.'
    return
}

if (-not (Test-Path -LiteralPath $taskPythonPath)) { throw 'First run .\scripts\run-chat.ps1 -Setup' }
$taskExistingSite = $null
try { $taskExistingSite = Invoke-RestMethod 'http://127.0.0.1:8004/api/status' -TimeoutSec 3 } catch { }
if ($taskExistingSite) {
    if (-not $taskExistingSite.method -or -not $taskExistingSite.model) { throw 'Another application is using port 8004. Preserve it and stop it yourself before launching EditGuard.' }
    Write-Host 'EditGuard is already running: http://127.0.0.1:8004/'
    if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:8004/' }
    return
}
Write-Host 'EditGuard: http://127.0.0.1:8004/ — keep this window open. Ctrl+C stops the website.'
if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:8004/' }
& $taskPythonPath -m editguard_chat.server
if ($LASTEXITCODE -ne 0) { throw 'The EditGuard server stopped with an error.' }
