# solve.ps1 — Task 2: Job Shop Scheduling - Verification

$SCRIPT_DIR = Split-Path -Parent $PSCommandPath
$TASK_DIR = Split-Path -Parent $SCRIPT_DIR

Write-Host "================================================"
Write-Host "  Task 2: Job Shop Scheduling - Verification"
Write-Host "================================================"
Write-Host ""

Push-Location $TASK_DIR

Write-Host "[1/2] Running black-box tests..."
python -m pytest solution/tests/test_env.py -v
if ($LASTEXITCODE -ne 0) { Pop-Location; exit 1 }
Write-Host ""

Write-Host "[2/2] Running quick training check..."
python solution/train.py
if ($LASTEXITCODE -ne 0) { Pop-Location; exit 1 }
Write-Host ""

Write-Host "================================================"
Write-Host "  All verification passed!"
Write-Host "================================================"

Pop-Location
exit 0
