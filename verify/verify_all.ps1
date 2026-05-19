# verify_all.ps1 — Run all task verification scripts sequentially
# Exit code 0 if all pass, 1 if any fail.

$ErrorActionPreference = "Stop"

$ROOT_DIR = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$FAILED = @()

$TASKS = @(
    "01_inventory_management",
    "02_job_scheduling",
    "03_ad_bidding",
    "04_cache_policy"
)

Write-Host "========================================"
Write-Host "  RL Benchmark - Full Verification"
Write-Host "========================================"
Write-Host ""

foreach ($task in $TASKS) {
    Write-Host ">>> [$task] Running verification..."
    Write-Host ""

    $SOLVE_SCRIPT = "$ROOT_DIR\tasks\$task\solution\solve.ps1"
    $TASK_DIR = "$ROOT_DIR\tasks\$task"

    if (-not (Test-Path $SOLVE_SCRIPT)) {
        Write-Host "!!! [$task] solve.ps1 not found, running commands directly"

        Push-Location $TASK_DIR

        try {
            Write-Host "[1/2] Running black-box tests..."
            python -m pytest solution/tests/test_env.py -v
            if ($LASTEXITCODE -ne 0) { throw "Tests failed" }

            Write-Host "[2/2] Running quick training check..."
            python solution/train.py
            if ($LASTEXITCODE -ne 0) { throw "Training failed" }
        } catch {
            Write-Host "!!! [$task] FAILED: $_"
            $FAILED += $task
        } finally {
            Pop-Location
        }
    } else {
        & $SOLVE_SCRIPT
        if ($LASTEXITCODE -ne 0) {
            Write-Host "!!! [$task] FAILED (exit code $LASTEXITCODE)"
            $FAILED += $task
        }
    }

    if ($task -notin $FAILED) {
        Write-Host ">>> [$task] PASSED"
    }

    Write-Host ""
    Write-Host "----------------------------------------"
    Write-Host ""
}

Write-Host "========================================"
Write-Host "  Summary"
Write-Host "========================================"
Write-Host ""

if ($FAILED.Count -eq 0) {
    Write-Host "All tasks passed. Exit code 0."
    exit 0
} else {
    Write-Host "Failed tasks: $($FAILED -join ', ')"
    Write-Host "Exit code 1."
    exit 1
}
