# Contributing to Likelihoodlum

Thanks for your interest in making LLM detection better! 🎲

## Quick Start

1. Fork the repo
2. Clone your fork
3. Make your changes
4. Test against a few repos (see [Testing](#testing))
5. Open a PR

## Adding a New Heuristic

This is the most impactful way to contribute. Here's how:

### 1. Define the Signal

Before writing code, answer these questions:

- **What does it measure?** (e.g. "comment density per file")
- **Why does it indicate LLM usage?** (e.g. "LLMs over-comment; humans are lazier")
- **What's the scoring range?** (follow the existing pattern: positive points for suspicious, negative for clearly human)
- **What are the false positive risks?**

### 2. Implement It

All scoring happens in `score_repo()` in `llm_detector.py`. Follow the existing pattern:

```python
# --- N. Your Signal Name (X-Y pts, can subtract up to -Z) ---
if some_condition:
    if very_suspicious:
        score += max_points
        reasons.append(f"Description of what was found [{max_points:+.0f}]")
    elif somewhat_suspicious:
        score += partial_points
        reasons.append(f"Milder description [{partial_points:+.0f}]")
    elif clearly_human:
        penalty = -Z
        score += penalty
        reasons.append(f"Human-like pattern [{penalty:+.0f}]")
```

Key principles:
- **Use `authored_total`** not `total_changes` — generated files should already be filtered out
- **Exclude bot authors** — use `is_bot_author()` where relevant
- **Include negative signals** — if your heuristic can identify clearly human patterns, subtract points
- **Always append to `reasons`** — every scoring decision should be explainable

### 3. Add Display Output

Update `print_report()` to show the raw data for your signal, and update the JSON output in `main()` if applicable.

### 4. Update the README

- Add your signal to the **Scoring Signals** table
- Add any new thresholds to the relevant threshold tables
- Update the point ranges in the Note

## Testing

Always test against a spread of repos:

```bash
# Known LLM-generated (should score high)
python3 llm_detector.py anthropics/claudes-c-compiler

# Known human multi-contributor (should score low)
python3 llm_detector.py django/django
python3 llm_detector.py golang/go

# Known human single-contributor (trickier — should still score low)
python3 llm_detector.py some/solo-human-project

# Your target repo
python3 llm_detector.py owner/repo
```

**Golden rule:** Don't inflate scores on known-human repos just to catch one more LLM repo. False positives are worse than false negatives for a tool like this.

## Code Style

- **Zero dependencies** — stdlib only. Don't add `requests`, `numpy`, or anything else.
- **Type hints** — use them for function signatures
- **Docstrings** — every function gets one
- **f-strings** — for all string formatting

## Reporting False Positives / Negatives

Found a repo that scores wrong? Open an issue with:

1. The repo URL
2. The score it got
3. What you think the score should be
4. Why (what the tool is getting wrong)

This is extremely valuable — it helps us tune thresholds and find blind spots.

## Ideas We'd Love PRs For

- **Diff complexity / entropy analysis** — are the diffs structured or chaotic?
- **File-type breakdown** — LLMs love generating configs and boilerplate
- **Comment density analysis** — LLMs over-explain; humans under-explain
- **Code style consistency** — LLMs are eerily consistent across files
- **Cross-file similarity** — LLMs repeat patterns; humans get creative (or sloppy)
- **Language-specific tuning** — different languages have different "normal" velocities
- **Commit time-of-day analysis** — 4am coding sessions hitting 500 lines/hr?

## License

By contributing, you agree that your contributions will be licensed under the MIT License.