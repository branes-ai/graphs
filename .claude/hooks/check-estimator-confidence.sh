#!/bin/bash
# PostToolUse hook: Verify that new/modified estimator code includes confidence tracking.
# Fires after Edit or Write on estimation/ files.
# Exit 0 = OK, Exit 2 = block with message.

INPUT=$(cat)
FILE=$(echo "$INPUT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
ti = data.get('tool_input', {})
print(ti.get('file_path', ti.get('content', '')))
" 2>/dev/null)

# Only check estimation/ Python files
case "$FILE" in
  */estimation/*.py)
    # Check if the file contains confidence-related imports or usage
    if [ -f "$FILE" ]; then
      if grep -q "class.*Analyzer" "$FILE" 2>/dev/null; then
        if ! grep -q "confidence\|ConfidenceLevel\|EstimationConfidence" "$FILE" 2>/dev/null; then
          echo "WARNING: Estimator file $FILE does not reference confidence tracking." >&2
          echo "All estimators should include EstimationConfidence in their descriptors." >&2
          echo "See core/confidence.py for ConfidenceLevel enum." >&2
          # Warn but don't block (exit 0)
          exit 0
        fi
      fi
    fi
    ;;
esac

exit 0
