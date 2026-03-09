#!/bin/bash
# PostToolUse hook: Syntax-check Python files after Write/Edit.
# Exit 0 = OK, Exit 2 = block with error.

INPUT=$(cat)
FILE=$(echo "$INPUT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('tool_input', {}).get('file_path', ''))
" 2>/dev/null)

case "$FILE" in
  *.py)
    if [ -f "$FILE" ]; then
      ERROR=$(python3 -m py_compile "$FILE" 2>&1)
      if [ $? -ne 0 ]; then
        echo "Syntax error in $FILE:" >&2
        echo "$ERROR" >&2
        exit 2
      fi
    fi
    ;;
esac

exit 0
