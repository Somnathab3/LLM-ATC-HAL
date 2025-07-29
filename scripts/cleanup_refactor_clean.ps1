# PowerShell version of cleanup_refactor.sh
Write-Host "Removing placeholder comments..." -ForegroundColor Green
Get-ChildItem -Path . -Filter "*.py" -Recurse | ForEach-Object {
    try {
        (Get-Content $_.FullName) | Where-Object { $_ -notmatch "TODO: remove" } | Set-Content $_.FullName
    } catch {
        Write-Warning "Could not process file: $($_.FullName)"
    }
}

Write-Host "Deleting empty or legacy files..." -ForegroundColor Green
@("llm_atc\agents", "llm_atc\baseline_models", "llm_atc\memory", "llm_atc\metrics", "llm_atc\tools", "llm_atc\data") | ForEach-Object {
    if (Test-Path $_) {
        Get-ChildItem -Path $_ -File -Recurse | Where-Object { $_.Length -eq 0 } | Remove-Item -Force -ErrorAction SilentlyContinue
    }
}

Get-ChildItem -Path . -Filter "*_old*.py" -Recurse | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Filter "*_backup*.py" -Recurse | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Filter "*milvus*.py" -Recurse | Remove-Item -Force -ErrorAction SilentlyContinue

if (Test-Path "docs\notebooks") {
    Get-ChildItem -Path "docs\notebooks" -Filter "*.ipynb" -Recurse | Where-Object { $_.Length -eq 0 } | Remove-Item -Force -ErrorAction SilentlyContinue
}

Write-Host "Clearing caches..." -ForegroundColor Green
Get-ChildItem -Path . -Name "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Filter "*.pyc" -Recurse | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Filter "*~" -Recurse | Remove-Item -Force -ErrorAction SilentlyContinue
if (Test-Path ".pytest_cache") { Remove-Item -Path ".pytest_cache" -Recurse -Force -ErrorAction SilentlyContinue }

Write-Host "Cleanup complete." -ForegroundColor Green
