#!/usr/bin/env bash
#
# CI check: ensure no new code reads measurement JSON files directly.
# All measurement loading must go through GroundTruthLoader.
#
# Safe-listed files that legitimately read JSON directly:
#   - ground_truth.py          (IS the loader)
#   - migrate_measurements.py  (migration tool, reads old format by design)
#   - migrate_calibration_structure.py (migration tool)
#   - aggregate_efficiency.py  (legacy path deprecated, will be removed)
#   - gpu_calibration.py       (reads efficiency_curves.json, not measurements)
#   - calibrate_efficiency.py  (reads efficiency_curves.json, not measurements)
#   - run_full_calibration.py  (reads efficiency_curves.json for validation)
#   - query_calibration_data.py (reads efficiency_curves.json for display)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

SAFELIST=(
    "ground_truth.py"
    "migrate_measurements.py"
    "migrate_calibration_structure.py"
    "cleanup_calibration_data.py"
    "aggregate_efficiency.py"
    "gpu_calibration.py"
    "calibrate_efficiency.py"
    "run_full_calibration.py"
    "query_calibration_data.py"
)

# Build grep exclusion pattern
EXCLUDE_ARGS=""
for f in "${SAFELIST[@]}"; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude=$f"
done

# Pattern: json.load( on measurement-like paths
# We look for load_measurement_file, json.load combined with measurement-related
# path patterns, and direct glob for .json in measurement directories.
FOUND=0

echo "Checking for direct measurement JSON reads outside GroundTruthLoader..."
echo

# Check 1: load_measurement_file() calls outside safe-listed files
if grep -rn "load_measurement_file" \
    --include="*.py" \
    $EXCLUDE_ARGS \
    "$REPO_ROOT/cli/" "$REPO_ROOT/src/graphs/" 2>/dev/null; then
    echo "ERROR: Found load_measurement_file() usage outside safe-listed files"
    FOUND=1
fi

# Check 2: json.load() near measurement/calibration_data path references
if grep -rn 'json\.load' \
    --include="*.py" \
    $EXCLUDE_ARGS \
    "$REPO_ROOT/cli/" "$REPO_ROOT/src/graphs/" 2>/dev/null \
    | grep -i 'measurement'; then
    echo "ERROR: Found json.load() with measurement-related context"
    FOUND=1
fi

# Check 3: glob("*.json") on measurement directories
if grep -rn 'glob.*measurement.*json\|measurements.*glob.*json' \
    --include="*.py" \
    $EXCLUDE_ARGS \
    "$REPO_ROOT/cli/" "$REPO_ROOT/src/graphs/" 2>/dev/null; then
    echo "ERROR: Found glob for JSON files in measurement directories"
    FOUND=1
fi

# Check 4: old measurements/ directory should not exist
if [ -d "$REPO_ROOT/measurements" ]; then
    COUNT=$(find "$REPO_ROOT/measurements" -name "*.json" | wc -l)
    if [ "$COUNT" -gt 0 ]; then
        echo "ERROR: Old measurements/ directory still contains $COUNT JSON files"
        echo "       Run: git rm -r measurements/"
        FOUND=1
    fi
fi

if [ "$FOUND" -eq 0 ]; then
    echo "OK: No legacy measurement JSON reads found"
fi

exit $FOUND
