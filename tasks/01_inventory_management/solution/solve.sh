#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "================================================"
echo "  Task 1: Inventory Management - Verification"
echo "================================================"
echo ""

# Step 1: Run black-box tests
echo "[1/2] Running black-box tests..."
cd "$TASK_DIR"
python -m pytest solution/tests/test_env.py -v
echo ""

# Step 2: Quick training verification
echo "[2/2] Running quick training check..."
cd "$TASK_DIR"
python solution/train.py
echo ""

echo "================================================"
echo "  All verification passed!"
echo "================================================"
