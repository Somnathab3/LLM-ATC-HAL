# PowerShell version of sanity_run.sh
Write-Host "Running unit tests..." -ForegroundColor Green
C:/Users/Administrator/AppData/Local/Programs/Python/Python313/python.exe -m pytest tests/test_modules.py -q

Write-Host "Running demo CLI..." -ForegroundColor Green
C:/Users/Administrator/AppData/Local/Programs/Python/Python313/python.exe -m llm_atc.cli demo

Write-Host "Running tiny shift benchmark..." -ForegroundColor Green
C:/Users/Administrator/AppData/Local/Programs/Python/Python313/python.exe -m llm_atc.cli shift-benchmark --config experiments/shift_experiment_config.yaml --tiers in_distribution --n 3

Write-Host "Sanity run completed without errors." -ForegroundColor Green
