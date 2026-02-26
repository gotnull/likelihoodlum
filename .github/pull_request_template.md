## What does this PR do?

<!-- A clear, concise description of the change. -->

## Why?

<!-- What problem does it solve? Link to an issue if applicable. -->

Closes #

## Type of change

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New heuristic / signal
- [ ] 🔧 Threshold tuning
- [ ] 📝 Documentation
- [ ] 🏗️ Refactor (no behavior change)
- [ ] 🧪 Tests

## New Heuristic Checklist

_Skip this section if not adding a new scoring signal._

- [ ] Defined in `score_repo()` following the existing pattern
- [ ] Includes both positive and negative signals where applicable
- [ ] Uses `authored_total` (not `total_changes`) for line counts
- [ ] Excludes bot authors via `is_bot_author()`
- [ ] Appends to `reasons` with point values shown (e.g. `[+5]`, `[-10]`)
- [ ] Added display output in `print_report()`
- [ ] Added to JSON output in `main()` (if applicable)
- [ ] Updated README scoring signals table
- [ ] Updated CHANGELOG.md

## Testing

Tested against these repos (paste relevant output):

**Should score HIGH (LLM-generated):**
```
python3 llm_detector.py anthropics/claudes-c-compiler --max-commits 50
```
<!-- Paste score line -->

**Should score LOW (human-written):**
```
python3 llm_detector.py django/django --max-commits 50
```
<!-- Paste score line -->

**Target repo (if applicable):**
```
python3 llm_detector.py owner/repo
```
<!-- Paste score line -->

## Does this change affect existing scores?

<!-- If yes, list the before/after for key repos. E.g.:
- django/django: 0 → 0 (unchanged)
- claudes-c-compiler: 81 → 83 (+2)
-->