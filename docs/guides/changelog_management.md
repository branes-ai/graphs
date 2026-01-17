# CHANGELOG Management Strategy

**Problem**: CHANGELOG.md has grown too large (>25000 tokens) for AI assistants to read and regain context efficiently.

**Solution**: Dual-file approach with automatic archival policy.

---

## File Structure

### `CHANGELOG.md` (Full History)
- **Purpose**: Complete, canonical project history
- **Audience**: Human developers, historical reference
- **Retention**: Forever
- **Format**: Standard Keep a Changelog format

### `CHANGELOG_RECENT.md` (Last 3 Months)
- **Purpose**: Quick context for AI assistants resuming work
- **Audience**: AI assistants (Claude Code, etc.)
- **Retention**: Last 3 months of entries (~10-20 entries max)
- **Token Target**: <15000 tokens (well within Read tool limits)
- **Format**: Same as CHANGELOG.md but condensed

---

## Workflow for Developers

### Adding New Entries

**Always add to both files:**

1. Add full entry to `CHANGELOG.md` (top of file, after Unreleased)
2. Add same entry to `CHANGELOG_RECENT.md` (top of file)

**Example**:
```bash
# After making changes, document them:
vim CHANGELOG.md          # Add detailed entry
vim CHANGELOG_RECENT.md   # Add same entry
```

### Monthly Archival (First of Month)

Every month, clean up `CHANGELOG_RECENT.md`:

```bash
# 1. Check age of oldest entry
tail -50 CHANGELOG_RECENT.md

# 2. Remove entries older than 3 months
#    - Keep entries from current month
#    - Keep entries from previous 2 months
#    - Remove everything older

# 3. Update "Last Updated" date at top of CHANGELOG_RECENT.md
```

**Example**:
```
Today: 2025-10-25
Keep: 2025-10-xx, 2025-09-xx, 2025-08-xx
Remove: 2025-07-xx and older
```

---

## Workflow for AI Assistants

### Starting a New Session

**Context Regaining Strategy**:

1. **First, read** `CHANGELOG_RECENT.md` (always fits in token budget)
2. **Then, read** session logs if referenced: `docs/sessions/YYYY-MM-DD_*.md`
3. **Optionally, read** `CHANGELOG.md` with offset if specific historical context needed

**Example**:
```python
# AI reads these files in order:
1. CHANGELOG_RECENT.md              # Quick context (<15000 tokens)
2. docs/sessions/2025-10-25_*.md    # Last session details
3. CLAUDE.md                        # Project structure (if needed)
```

### During Active Work

**Update both files** when making significant changes:
- Add entry to `CHANGELOG.md` (full detail)
- Add entry to `CHANGELOG_RECENT.md` (same content)

### Creating Session Logs

At end of session:
```python
# Create session log
docs/sessions/YYYY-MM-DD_topic.md

# Reference it in CHANGELOG_RECENT.md
"See docs/sessions/YYYY-MM-DD_topic.md for details"
```

---

## Token Budget Guidelines

### Target Sizes

| File | Token Target | Current | Status |
|------|-------------|---------|--------|
| `CHANGELOG_RECENT.md` | <15000 | ~5000 | ✅ Good |
| `CHANGELOG.md` | Unlimited | ~30000 | ⚠ Use offset |
| Session logs | <10000 | ~3000 | ✅ Good |

### When CHANGELOG_RECENT.md Gets Too Large

**If approaching 20000 tokens:**
1. Archive oldest month immediately (don't wait for monthly cycle)
2. Consider condensing entries (remove implementation details, keep impact)
3. Reference session logs for details: "See docs/sessions/... for implementation"

---

## Benefits

### For Humans
✅ Complete history preserved in `CHANGELOG.md`
✅ Standard Keep a Changelog format
✅ Easy to browse full project evolution

### For AI Assistants
✅ Quick context regaining (<15000 tokens)
✅ Always readable in single Read operation
✅ References to session logs for deep dives
✅ No need to use offset/pagination

### For Project
✅ Better documentation hygiene
✅ Forced regular review of changes
✅ Encourages detailed session logs
✅ Maintains both human and AI accessibility

---

## Automation Opportunities

### Shell Script (Optional)

```bash
#!/bin/bash
# archive_changelog.sh - Archive old CHANGELOG_RECENT.md entries

# Get date 3 months ago
CUTOFF_DATE=$(date -d "3 months ago" +%Y-%m)

# Extract entries newer than cutoff from CHANGELOG_RECENT.md
# (Implementation left as exercise - requires careful markdown parsing)

echo "Archived entries older than $CUTOFF_DATE"
```

### Git Hook (Optional)

```bash
# .git/hooks/pre-commit
# Warn if CHANGELOG_RECENT.md is getting large

RECENT_SIZE=$(wc -c < CHANGELOG_RECENT.md)
if [ $RECENT_SIZE -gt 80000 ]; then
    echo "⚠ CHANGELOG_RECENT.md is large ($RECENT_SIZE bytes)"
    echo "  Consider archiving old entries"
fi
```

---

## FAQs

**Q: Why not just use CHANGELOG.md with offset?**
A: Reading with offset requires knowing exact line ranges and multiple Read calls. CHANGELOG_RECENT.md is optimized for single-read context regaining.

**Q: Why 3 months retention?**
A: Balances recent context (typically enough) with token budget. Adjustable if needed.

**Q: What if I need older context?**
A: AI can still read `CHANGELOG.md` with offset, or search session logs in `docs/sessions/`.

**Q: Should session logs be archived too?**
A: No. Session logs are already timestamped and self-contained. They can stay indefinitely.

**Q: What about unreleased changes?**
A: Keep [Unreleased] section in both files, archive when released.

---

## Current Status (2025-10-25)

- ✅ `CHANGELOG.md`: Complete history (~30000 tokens)
- ✅ `CHANGELOG_RECENT.md`: Created with last 5 entries (~5000 tokens)
- ✅ Session logs: 17 logs in `docs/sessions/`
- ✅ This guide: Created

**Next archival due**: 2025-11-01 (remove entries from 2025-07 and older)

---

## Example Workflow

### Scenario: AI Resuming Work After 1 Week

```python
# Step 1: Read recent context
read("CHANGELOG_RECENT.md")
# → Gets last 3 months of changes (~5000 tokens)

# Step 2: Read last session
read("docs/sessions/2025-10-25_hardware_tests_and_automotive_fix.md")
# → Gets detailed last session (~3000 tokens)

# Step 3: Ready to work
# Total context: ~8000 tokens (well within budget)
```

### Scenario: Developer Adds New Feature

```bash
# 1. Implement feature
vim src/graphs/hardware/new_feature.py

# 2. Document in both CHANGELOGs
vim CHANGELOG.md         # Add: [2025-10-26] - New Feature
vim CHANGELOG_RECENT.md  # Add: Same entry

# 3. Commit
git add CHANGELOG.md CHANGELOG_RECENT.md src/graphs/hardware/new_feature.py
git commit -m "Add new feature with CHANGELOG updates"
```

---

## Maintenance Checklist

**Monthly** (1st of month):
- [ ] Review `CHANGELOG_RECENT.md` size
- [ ] Archive entries older than 3 months
- [ ] Update "Last Updated" date
- [ ] Verify token count <15000

**Per Session** (AI work):
- [ ] Read `CHANGELOG_RECENT.md` at start
- [ ] Update both CHANGELOGs during work
- [ ] Create session log at end
- [ ] Reference session log in CHANGELOG_RECENT.md

**As Needed**:
- [ ] If CHANGELOG_RECENT.md >20000 tokens: Archive immediately
- [ ] If CHANGELOG.md structure changes: Update this guide
- [ ] If retention policy changes: Document in this guide
