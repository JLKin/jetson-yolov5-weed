param(
    [string]$PythonExe = "python",
    [switch]$ReuseSystemPackages = $true
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    if ($ReuseSystemPackages) { & $PythonExe -m venv --system-site-packages .venv }
    else { & $PythonExe -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw "Failed to create virtual environment" }
}
& .\.venv\Scripts\python.exe scripts/install_runtime.py
if ($LASTEXITCODE -ne 0) { throw "Installation failed; inspect the error above" }
Write-Host "Ready. Run: .\.venv\Scripts\python.exe -m weed_detector --help"
