#!/bin/bash
# PreToolUse hook: Prevent accidental deletion or overwriting of calibration profiles.
# Calibration data is expensive to produce -- require explicit confirmation.
# Exit 0 = OK, Exit 2 = block.

INPUT=$(cat)
TOOL=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_name',''))" 2>/dev/null)
FILE=$(echo "$INPUT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
ti = data.get('tool_input', {})
# Check file_path for Edit/Write, command for Bash
print(ti.get('file_path', ti.get('command', '')))
" 2>/dev/null)

case "$FILE" in
  */calibration/profiles/*.json)
    if [ "$TOOL" = "Write" ]; then
      echo "BLOCKED: Direct overwrite of calibration profile $FILE." >&2
      echo "Calibration profiles contain measured data. Use the calibration workflow instead." >&2
      echo "See docs/calibration_workflow.md for the correct procedure." >&2
      exit 2
    fi
    ;;
  *rm*calibration/profiles*)
    echo "BLOCKED: Attempting to delete calibration profiles." >&2
    echo "Calibration data is expensive to reproduce." >&2
    exit 2
    ;;
esac

exit 0
