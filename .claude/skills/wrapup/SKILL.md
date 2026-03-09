---
name: wrapup
description: Create a changelog entry and session log documenting work done in this session. Use at the end of a development session to record what was accomplished.
disable-model-invocation: true
---

# Session Wrapup

Generate documentation for this session's work.

## Steps

1. **Review what changed**: Examine git diff and recent commits to understand all modifications made in this session.

2. **Create changelog entry** in `docs/changelog/`:
   - File: `docs/changelog/YYYY-MM-DD.md` (today's date)
   - Format:
     ```markdown
     # Changelog YYYY-MM-DD

     ## [Category]
     - Description of change (files affected)
     ```
   - Categories: Added, Changed, Fixed, Deprecated, Removed, Performance, Infrastructure

3. **Create session log** in `docs/logs/`:
   - File: `docs/logs/YYYY-MM-DD_<short-description>.log`
   - Include: what was done, key decisions, files modified, tests run

4. **Update contribution log** in `.claude/contributions/`:
   - Append to current month's JSONL file (`.claude/contributions/YYYY-MM.jsonl`)
   - Format: `{"date": "YYYY-MM-DD", "summary": "...", "files_changed": N, "lines_added": N, "lines_removed": N}`

5. **Create decision record** if architectural decisions were made:
   - Copy `.claude/decisions/template.yaml` to `DECISION-YYYY-MM-DD-NNN.yaml`
   - Fill in summary, trigger, alternatives, impact, verification
