# Documentation System Guide

This guide explains how to track and document work on this project.

---

## 📚 Documentation Structure

```
graphs/
├── SUMMARY.md                    # 🎯 START HERE - Current project state
├── CHANGELOG.md                  # 📝 Daily updates log
├── DOCUMENTATION_GUIDE.md        # 📖 This file
├── CLAUDE.md                     # 🤖 Instructions for AI assistant
├── docs/
│   ├── sessions/                 # 📅 Detailed session logs
│   │   ├── README.md            # How to use sessions
│   │   ├── template.md          # Template for new sessions
│   │   ├── 2025-10-20_fusion_partitioning.md
│   │   └── 2025-10-19_graph_partitioning.md
│   ├── GETTING_STARTED.md       # 🚀 User quick start
│   ├── graph_partitioner_tutorial.md
│   ├── realistic_performance_modeling_plan.md  # 🗺️ Master plan
│   └── ...                      # Technical documentation
└── ...
```

---

## 📖 Documentation Hierarchy

### Level 1: Quick Overview
**File**: `SUMMARY.md`
**Purpose**: High-level project status and accomplishments
**Update**: After major milestones
**Read time**: 10-15 minutes

**Contents**:
- Current status (what's done, what's in progress)
- What we built (capabilities overview)
- Key results and validation
- Quick reference commands
- Roadmap

**When to read**:
- Morning start (quick refresh)
- After returning from break
- Reviewing project status

### Level 2: Daily Updates
**File**: `CHANGELOG.md`
**Purpose**: Chronological log of changes
**Update**: End of each work session
**Read time**: 2-5 minutes per day

**Contents**:
- Date-stamped entries
- What was added/changed/fixed
- Key results and metrics
- References to detailed session logs

**When to read**:
- Daily standup review
- Weekly planning
- Looking for specific changes

### Level 3: Detailed Sessions
**Directory**: `docs/sessions/YYYY-MM-DD_topic.md`
**Purpose**: In-depth session documentation
**Update**: End of each significant session
**Read time**: 20-30 minutes per session

**Contents**:
- Goals and accomplishments
- Implementation details
- Code snippets and examples
- Challenges and solutions
- Next steps and open questions

**When to read**:
- Picking up where you left off
- Debugging specific features
- Understanding design decisions
- Onboarding review

### Level 4: Technical Docs
**Directory**: `docs/` (various files)
**Purpose**: Deep technical documentation
**Update**: As features are completed
**Read time**: 30-60 minutes per doc

**Contents**:
- Architecture designs
- Algorithm specifications
- API documentation
- Validation reports

**When to read**:
- Implementing new features
- Understanding subsystem details
- Research and planning

---

## 🔄 Daily Workflow

### At Start of Work Session

1. **Quick refresh** (5 minutes):
   ```bash
   # Read current status
   cat SUMMARY.md | head -50

   # Check recent changes
   cat CHANGELOG.md | head -100

   # Review last session
   cat docs/sessions/$(ls -t docs/sessions/*.md | head -1)
   ```

2. **Plan today's work**:
   - Check "Next Steps" from last session summary
   - Review any open questions
   - Set goals for today

### During Work Session

3. **Take notes** as you work:
   - Accomplishments
   - Key insights
   - Challenges encountered
   - Decisions made
   - Useful code snippets

### At End of Work Session

4. **Document the session** (15-20 minutes):

   a. **Create session summary**:
   ```bash
   # Copy template
   cp docs/sessions/template.md docs/sessions/$(date +%Y-%m-%d)_topic.md

   # Fill in the details
   vim docs/sessions/$(date +%Y-%m-%d)_topic.md
   ```

   b. **Update CHANGELOG.md**:
   ```markdown
   ## [YYYY-MM-DD] - Brief Description

   ### Added
   - Feature 1 with results
   - Feature 2 with metrics

   ### Changed
   - Modification 1

   ### Fixed
   - Bug 1
   ```

   c. **Update SUMMARY.md** (if major milestone):
   - Update status section
   - Add new results
   - Update roadmap

5. **Quick review checklist**:
   - [ ] Session summary created and complete
   - [ ] CHANGELOG.md entry added
   - [ ] SUMMARY.md updated (if needed)
   - [ ] Next steps clearly defined
   - [ ] Code committed (if applicable)

---

## 📅 Weekly Review

Every week (or major milestone):

1. **Review the week** (30 minutes):
   ```bash
   # Get this week's session summaries
   ls -lt docs/sessions/*.md | head -7

   # Review CHANGELOG for the week
   cat CHANGELOG.md | head -200
   ```

2. **Update planning**:
   - What went well?
   - What challenges did we face?
   - Are we on track for phase goals?
   - What should we adjust?

3. **Update SUMMARY.md**:
   - Refresh current status
   - Update metrics and statistics
   - Revise roadmap if needed

4. **Clean up**:
   - Archive old notes
   - Update stale documentation
   - Consolidate related session summaries

---

## 🎯 Use Cases

### "I'm starting work today. What should I do?"

```bash
# 1. Quick 5-minute refresh
cat SUMMARY.md | head -50

# 2. Check yesterday's work
ls -t docs/sessions/*.md | head -1 | xargs cat

# 3. See what's next
# Look at "Next Steps" section from latest session
```

### "We had a productive day! How do I document it?"

```bash
# 1. Create session summary from template
cp docs/sessions/template.md docs/sessions/$(date +%Y-%m-%d)_topic.md

# 2. Fill in all sections (takes ~15 minutes)

# 3. Add entry to CHANGELOG.md

# 4. Update SUMMARY.md if major milestone
```

### "What happened last week?"

```bash
# 1. Read CHANGELOG entries for the week
cat CHANGELOG.md | head -200

# 2. Skim session summaries
ls -lt docs/sessions/*.md | head -7

# 3. Check updated metrics in SUMMARY.md
```

### "How do I onboard someone new?"

Reading order:
1. `SUMMARY.md` - Get the big picture (10-15 min)
2. `CHANGELOG.md` - See recent progress (5-10 min)
3. `docs/realistic_performance_modeling_plan.md` - Understand architecture (30-60 min)
4. Recent session summaries - Understand current work (30-60 min)
5. `docs/GETTING_STARTED.md` - Try running code (30 min)

### "I need to debug/understand a feature"

```bash
# 1. Find when it was implemented
grep -r "feature name" docs/sessions/*.md

# 2. Read that session summary for context

# 3. Check related documentation
# (listed in session summary references)

# 4. Review code with context
```

---

## ✅ Documentation Quality Checklist

### For Session Summaries

Good session summary has:
- [ ] Specific goals listed
- [ ] Concrete accomplishments described
- [ ] Metrics/numbers included (not just "improved performance")
- [ ] Key insights captured (what you learned)
- [ ] Files created/modified with line counts
- [ ] Next steps clearly defined
- [ ] Challenges and solutions documented
- [ ] Code snippets for key concepts

### For CHANGELOG Entries

Good changelog entry has:
- [ ] Date in format `[YYYY-MM-DD]`
- [ ] Brief descriptive title
- [ ] Categorized changes (Added/Changed/Fixed)
- [ ] Specific results and metrics
- [ ] References to detailed session summaries

### For SUMMARY.md Updates

Good summary update includes:
- [ ] Updated status section (✅/🚧/📋 symbols)
- [ ] New validation results with numbers
- [ ] Updated metrics and statistics
- [ ] Revised roadmap if needed
- [ ] Updated "Last Updated" date

---

## 🚫 Common Pitfalls

### ❌ Don't Do This

**Vague entries**:
```markdown
## [2025-10-21]
- Fixed some bugs
- Improved performance
- Updated docs
```

**Missing details**:
```markdown
Implemented hardware mapper. It works.
```

**No follow-up**:
```markdown
Next steps: TBD
Open questions: Will figure out later
```

### ✅ Do This Instead

**Specific entries**:
```markdown
## [2025-10-21] - GPU Hardware Mapper Implementation

### Added
- GPU hardware mapper (src/graphs/characterize/gpu_mapper.py, 450 lines)
  - Maps fused subgraphs to SM groups
  - Accounts for wave quantization
  - Results: ResNet-18 uses 24/132 SMs (18% utilization) at batch=1

### Fixed
- Off-by-one error in fusion boundary detection (line 234)
  - Was causing incorrect subgraph counts (61 instead of 32)
  - Now validated: ResNet-18 correctly shows 32 fused subgraphs
```

**Detailed explanations**:
```markdown
Implemented GPU hardware mapper using wave-based allocation:

**Algorithm**:
1. Group subgraphs by execution stage (from concurrency analysis)
2. For each stage, calculate thread requirements per subgraph
3. Map threads to warps (32 threads/warp), then to SMs
4. Account for wave quantization (SMs allocated in waves of 4)

**Results**:
- ResNet-18: 32 subgraphs → 24 SMs utilized (18%)
- Fixes 1000× latency overestimate from naive peak calculation
```

**Clear next steps**:
```markdown
### Next Steps

**Immediate** (tomorrow):
1. [ ] Add KPU mapper (similar to GPU, but with tile constraints)
2. [ ] Test hardware mapper on MobileNet-V2 and EfficientNet-B0
3. [ ] Validate SM allocation against occupancy calculator

**This Week**:
1. [ ] Complete TPU and CPU mappers
2. [ ] Create test_hardware_mapping.py validation script
3. [ ] Update latency estimates using realistic utilization

**Open Questions**:
1. Should we account for dynamic SM allocation? (GPU can reassign)
2. How to handle heterogeneous execution (some ops memory-bound, some compute-bound)?
```

---

## 🔧 Tips & Best Practices

### Writing Session Summaries

1. **Write as you go**: Keep notes during work, then organize at end
2. **Be specific**: "42% memory reduction" not "better memory usage"
3. **Include context**: Why decisions were made, not just what was done
4. **Link everything**: Reference other docs, sessions, code locations
5. **Capture insights**: What surprised you? What did you learn?

### Maintaining Documentation

1. **Update daily**: Don't let documentation lag
2. **Review weekly**: Keep SUMMARY.md current
3. **Archive old**: Move completed work to archives if needed
4. **Stay consistent**: Use templates and checklists
5. **Be honest**: Document failures and challenges, not just successes

### Using Documentation

1. **Read before coding**: Review context first
2. **Update after coding**: Document while fresh
3. **Reference frequently**: Use docs to stay on track
4. **Share liberally**: Make it easy for others to follow along

---

## 📊 Example Timeline

### Day 1: Starting Hardware Mapping

**Morning** (9:00 AM):
```bash
# Quick refresh (5 min)
cat SUMMARY.md | head -50
cat docs/sessions/2025-10-20_fusion_partitioning.md | tail -50

# Start new session notes file
vim session_notes.txt
```

**During day**:
- Take notes in `session_notes.txt`
- Capture key insights as they happen
- Save code snippets that matter

**Evening** (5:00 PM):
```bash
# Create session summary (15 min)
cp docs/sessions/template.md docs/sessions/2025-10-21_gpu_hardware_mapper.md
vim docs/sessions/2025-10-21_gpu_hardware_mapper.md
# (Fill in from session_notes.txt)

# Update CHANGELOG (5 min)
vim CHANGELOG.md
# (Add today's entry at top)

# Commit
git add .
git commit -m "Day 1 GPU hardware mapper: SM allocation algorithm"
```

### Friday: Week Review

**End of week** (Friday 4:00 PM):
```bash
# Review the week (30 min)
ls -lt docs/sessions/*.md | head -7  # This week's sessions
cat CHANGELOG.md | head -300         # This week's changes

# Update SUMMARY.md (15 min)
vim SUMMARY.md
# - Update status section
# - Add this week's results
# - Update roadmap

# Plan next week (15 min)
vim docs/sessions/2025-10-24_weekly_plan.md
# - Review progress vs goals
# - Set next week's objectives
# - Identify blockers
```

---

## 🎓 Learning from Documentation

Good documentation helps you:

1. **Remember context**: "Why did we choose this approach?"
2. **Track progress**: "How much have we accomplished?"
3. **Avoid rework**: "We already tried that, it didn't work because..."
4. **Make decisions**: "Last time we chose X over Y because Z"
5. **Onboard others**: "Here's what we've done and why"
6. **Debug faster**: "This worked before, what changed?"

---

## 🤝 Collaboration

When working with others:

1. **Read before meetings**: Review recent sessions
2. **Reference in discussions**: "See session 2025-10-20, section 3"
3. **Update after decisions**: Document consensus and rationale
4. **Share context**: Point team members to relevant docs
5. **Stay synchronized**: Keep documentation up-to-date

---

## ❓ FAQ

**Q: How much time should I spend documenting?**
A: ~15-20 minutes at end of session. Worth it for future productivity!

**Q: What if I didn't accomplish much today?**
A: Still document! Challenges and learnings are valuable.

**Q: Should I document failed approaches?**
A: YES! "We tried X, it didn't work because Y" saves future time.

**Q: How detailed should session summaries be?**
A: Enough so that future-you (or someone else) can understand what happened and why.

**Q: What if I forget to document?**
A: Create summary next day while still fresh. Better late than never!

**Q: How do I handle long experiments?**
A: Create session summaries at natural breakpoints, even if work continues.

---

## 🎯 Summary

### The System
- **SUMMARY.md**: Current project state (update after milestones)
- **CHANGELOG.md**: Daily updates (update every session)
- **docs/sessions/**: Detailed logs (create every session)
- **docs/**: Technical documentation (update as features complete)

### The Workflow
1. **Start**: Read SUMMARY.md + last session
2. **Work**: Take notes as you go
3. **End**: Document session (15-20 min)
4. **Weekly**: Review and update SUMMARY.md

### The Benefits
- Never lose context
- Track progress clearly
- Make better decisions
- Onboard others easily
- Debug faster

---

**Questions?** Check out:
- `docs/sessions/README.md` - More on session summaries
- `docs/sessions/template.md` - Template with all sections
- Recent session summaries - Real examples

Happy documenting! 📝
