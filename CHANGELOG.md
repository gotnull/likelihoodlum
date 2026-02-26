# Changelog

All notable changes to Likelihoodlum are documented here.

## [1.0.0] — 2026-02-26

### The Public Release 🎲

First public release with nine heuristic signals and a comprehensive scoring engine.

### Scoring Signals

- **Code Velocity** (−10 to +35 pts) — Lines/min between consecutive commits by the same author, with trimmed mean boost when heavy tail detected
- **Session Productivity** (−5 to +20 pts) — Aggregate lines/min per coding session (>2hr gap = new session)
- **Commit Size Uniformity** (−5 to +15 pts) — Coefficient of variation in commit sizes; LLMs are suspiciously uniform
- **Commit Message Patterns** (0 to +15 pts) — Regex matching for LLM-typical phrasings, conventional commit verbosity, multi-scopes
- **Burst Detection** (0 to +15 pts) — Rapid bursts (>300 lines in <30 min) and sustained high-velocity sessions
- **Multi-Author Discount** (−10 to +5 pts) — More authors = less likely LLM; solo author = small bump
- **Extreme Per-Commit Velocity** (0 to +10 pts) — Intervals exceeding 50 lines/min (~3,000 lines/hr)
- **Project-Scale Plausibility** (−5 to +20 pts) — Total authored output vs repo creation date and active days
- **Generated File Ratio** (informational) — Percentage of changes in vendor/generated files

### Features

- Zero dependencies — stdlib-only Python 3.10+
- Concurrent commit detail fetching (up to 10 parallel requests)
- Bot author filtering (dependabot, renovate, etc.) — excluded from analysis and API calls
- Repo metadata fetch for true creation date
- Generated/vendored file filtering (lockfiles, protobufs, Xcode, build artifacts, assets)
- Bidirectional scoring — human patterns actively reduce the score
- Full JSON output mode (`--json`)
- `.env` file support with built-in fallback parser

### Documentation

- Wall of Truth leaderboard with 20+ tested repos
- Anthropic Spotlight section
- Contributing guide with heuristic development protocol
- Issue templates for bug reports and feature requests
- pip-installable via `pyproject.toml`

## [0.3.0] — 2026-02-26

### Added

- Project-scale plausibility heuristic comparing total output against repo creation date
- New `📈 Project-scale output` section in report showing repo creation date, active days, and daily output
- Daily output thresholds (300 / 800 / 2,000 / 5,000 lines per active day)
- `project_scale` object in JSON output

## [0.2.0] — 2026-02-26

### Added

- Conventional commit pattern detection (multi-scopes, verbose descriptions, `feat(a, b):`)
- Bot author filtering — `dependabot[bot]` and similar excluded from velocity, sessions, and author counts
- Trimmed mean velocity boost when trimmed mean >> median (heavy tail signal)
- `suspicious_pct` fallback for interval percentage scoring
- High-velocity session detection alongside duration-based bursts
- Per-commit extreme velocity signal (>50 lines/min)
- Clean-messages-but-high-velocity cross-signal
- Expanded LLM message patterns (`enhance X with Y`, `integrate/wire up`, `add X Y Z`)

### Changed

- Burst detection threshold lowered from 500 to 300 authored lines, minimum commits from 3 to 2
- Burst scoring bumped (3+ sessions: 8→10 pts)
- Velocity scoring now considers trimmed mean for tier boosting (22→26 or 30 pts)

## [0.1.0] — 2026-02-26

### Added

- Generated/vendored file filtering with `authored_total` vs `generated_total` breakdown
- Negative scoring signals (human velocity, human session pace, high commit size variation)
- Multi-author discount (−10 for 5+ authors, −5 for 3+)
- Solo author bonus (+5)
- Trimmed mean for velocity and session calculations
- `📁 Line changes breakdown` section in report
- Seven heuristic signals in initial scoring engine

## [0.0.1] — 2026-02-26

### Added

- Initial commit: basic LLM likelihood detector
- Five heuristic signals: velocity, sessions, bursts, commit size uniformity, message patterns
- GitHub API integration with token support
- CLI with `--token`, `--branch`, `--max-commits`, `--json` flags
- `.env` file support with python-dotenv fallback