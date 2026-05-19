#!/bin/bash
# verify_all.sh — Run all task verification scripts sequentially
# Exit code 0 if all pass, 1 if any fail.

set -e

TASKS=(
    "01_inventory_management"
    "02_job_scheduling"
    "03_ad_bidding"
    "04_cache_policy"
)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FAILED=()

echo "========================================"
echo "  RL Benchmark — Full Verification"
echo "========================================"
echo ""

for task in "${TASKS[@]}"; do
    echo ">>> [${task}] Running solve.sh..."
    echo ""

    SOLVE_SCRIPT="${ROOT_DIR}/tasks/${task}/solution/solve.sh"

    if [ ! -f "${SOLVE_SCRIPT}" ]; then
        echo "!!! [${task}] solve.sh not found, skipping"
        FAILED+=("${task}")
        echo ""
        continue
    fi

    bash "${SOLVE_SCRIPT}"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ">>> [${task}] PASSED"
    else
        echo "!!! [${task}] FAILED (exit code ${EXIT_CODE})"
        FAILED+=("${task}")
    fi

    echo ""
    echo "----------------------------------------"
    echo ""
done

echo "========================================"
echo "  Summary"
echo "========================================"
echo ""

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All tasks passed. Exit code 0."
    exit 0
else
    echo "Failed tasks: ${FAILED[*]}"
    echo "Exit code 1."
    exit 1
fi
