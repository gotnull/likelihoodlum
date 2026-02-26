#!/usr/bin/env python3
"""
LLM-Generated Code Detector (Likelihoodlum)

Analyzes a GitHub repository's commit history to estimate the likelihood
that the code was written by an LLM rather than a human.

Heuristics used:
  1. Code velocity: lines changed per minute between consecutive commits by
     the same author. Humans typically produce 10-30 LoC/hour on complex
     projects; LLM-assisted work can easily exceed 200+ LoC/hour.
  2. Burst commits: clusters of large commits made in rapid succession.
  3. Session analysis: groups consecutive commits into "coding sessions" and
     measures productivity per session.
  4. Commit size uniformity: LLM-generated commits tend to be uniformly large,
     while human commits vary more in size.
  5. Commit message analysis: generic / overly-perfect commit messages.
  6. Multi-author discount: real projects tend to have multiple contributors.
  7. Negative signals: clearly human velocity patterns actively reduce the score.

Generated / vendor files (protos, lockfiles, Xcode project files, etc.) are
filtered out so they don't inflate velocity measurements.

Usage:
    python3 llm_detector.py <github_repo_url_or_owner/repo> [--token GITHUB_TOKEN]
                                                              [--branch BRANCH]
                                                              [--max-commits N]
                                                              [--json]

Examples:
    python3 llm_detector.py https://github.com/owner/repo
    python3 llm_detector.py owner/repo --token ghp_xxxx --max-commits 500
"""

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:
    # Fallback: manually parse .env if python-dotenv is not installed
    def load_dotenv(dotenv_path=None, **kwargs):
        path = dotenv_path or Path(__file__).resolve().parent / ".env"
        path = Path(path)
        if path.is_file():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    os.environ.setdefault(key, value)


# Load .env from the same directory as this script
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# ---------------------------------------------------------------------------
# Constants & thresholds
# ---------------------------------------------------------------------------

# If two consecutive commits by the same author are more than this many minutes
# apart we consider them to be in different coding sessions.
SESSION_GAP_MINUTES = 120

# Lines-per-minute thresholds (applied to non-generated code only)
LPM_CLEARLY_HUMAN = 0.5  # ~30 LoC/hr – normal productive human
LPM_HUMAN_UPPER = 1.5  # ~90 LoC/hr – fast human, maybe some copy-paste
LPM_SUSPICIOUS = 4.0  # ~240 LoC/hr – quite fast, could be assisted
LPM_VERY_SUSPICIOUS = 10.0  # ~600 LoC/hr – almost certainly not hand-typed

# Daily output thresholds (authored lines per active day).
# A senior engineer writing production code typically produces 200-500 lines/day.
# These thresholds are deliberately generous to avoid false positives.
DAILY_OUTPUT_CLEARLY_HUMAN = 300  # Solid day of focused human work
DAILY_OUTPUT_HUMAN_UPPER = 800  # Very fast human, some scaffolding likely
DAILY_OUTPUT_SUSPICIOUS = 2000  # Hard to sustain without assistance
DAILY_OUTPUT_VERY_SUSPICIOUS = 5000  # Almost certainly not hand-written

# Minimum minutes between commits to consider for velocity calc.
# Commits less than 1 min apart are likely amend/rebase artifacts.
MIN_COMMIT_GAP_MINUTES = 1.0

# File extensions / patterns considered generated, vendored, or non-authored.
# Changes to these files are excluded from velocity calculations.
GENERATED_FILE_PATTERNS = [
    # Lock files & dependency manifests
    r"package-lock\.json$",
    r"yarn\.lock$",
    r"Podfile\.lock$",
    r"Gemfile\.lock$",
    r"composer\.lock$",
    r"Cargo\.lock$",
    r"poetry\.lock$",
    r"pnpm-lock\.yaml$",
    r"go\.sum$",
    # Xcode / Apple
    r"\.pbxproj$",
    r"\.xcworkspacedata$",
    r"\.xcscheme$",
    r"project\.pbxproj$",
    r"contents\.xcworkspacedata$",
    # Proto / generated
    r"\.pb\.go$",
    r"\.pb\.swift$",
    r"\.pb\.h$",
    r"\.pb\.cc$",
    r"\.pb\.m$",
    r"_pb2\.py$",
    r"_pb2_grpc\.py$",
    r"\.proto$",
    r"\.g\.dart$",
    r"\.freezed\.dart$",
    r"\.gen\.go$",
    r"\.generated\.",
    # Build artifacts & configs
    r"\.min\.js$",
    r"\.min\.css$",
    r"\.map$",
    r"dist/",
    r"build/",
    r"vendor/",
    r"node_modules/",
    r"\.DS_Store$",
    # Data / assets
    r"\.json$",  # most JSON is config/data, not hand-authored code
    r"\.svg$",
    r"\.png$",
    r"\.jpg$",
    r"\.ico$",
    r"\.woff",
    r"\.ttf$",
    r"\.eot$",
]

_GENERATED_RE = [re.compile(p, re.IGNORECASE) for p in GENERATED_FILE_PATTERNS]


def is_generated_file(filename: str) -> bool:
    """Return True if a filename looks like a generated/vendored file."""
    for pat in _GENERATED_RE:
        if pat.search(filename):
            return True
    return False


# Known bot author suffixes — excluded from author counts and velocity.
BOT_AUTHOR_SUFFIXES = ["[bot]", "-bot", "_bot"]

# Commit message patterns that are suspiciously "LLM-like".
# These are deliberately tighter than before – we want to catch the
# robotic-sounding messages, not normal dev shorthand.
LLM_MESSAGE_PATTERNS = [
    # Overly formal "Implement/Create" with a noun phrase
    r"^implement\s+\w+",
    r"^create\s+(the\s+)?\w+\s+(component|module|service|function|class|endpoint|handler|middleware|util)",
    # "Add X functionality/feature/support"
    r"^add\s+\w+\s+(functionality|feature|support|implementation|capability|module|component)",
    # Refactor/improve/enhance with a long description (LLMs love these)
    r"^(refactor|improve|enhance|optimize)\s+.{30,}",
    # "Update X to Y" with very specific phrasing
    r"^update\s+\w+\s+to\s+(handle|support|include|use|implement)",
    # Conventional commits with scopes — LLMs love over-specifying scopes
    r"^(feat|fix|docs|style|refactor|perf|test|chore)\(.{15,}\):",
    # Conventional commits with any scope + verbose description (>40 chars after colon)
    r"^(feat|fix|docs|style|refactor|perf|test|chore)\([^)]+\):\s+.{40,}",
    # Conventional commits with comma-separated or slash-separated multi-scopes
    r"^(feat|fix|docs|style|refactor|perf|test|chore)\([^)]*[,/][^)]*\):",
    # Messages that read like documentation
    r"^(ensure|make sure|modify)\s+.{20,}",
    # "Fix issue with X" or "Fix bug in X" — overly descriptive
    r"^fix\s+(issue|bug|problem|error)\s+(with|in|for|related\s+to)\s+",
    # Messages that list what was done (LLMs love bullet-point style)
    r"^(add|implement|create|update|fix)\s+.*\s+and\s+(add|implement|create|update|fix)\s+",
    # "Add <noun> <noun>" — short but formulaic (e.g. "Add user authentication")
    r"^add\s+\w+\s+\w+\s+\w+",
    # Descriptive verb phrases that sound like task descriptions
    r"^(integrate|wire up|set up|hook up|connect|configure)\s+.{15,}",
    # "Enhance X with Y" — very LLM-y phrasing
    r"^enhance\s+\w+\s+with\s+",
]

# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

# Max parallel requests when fetching commit details.
# Stay well under GitHub's abuse-detection limits.
MAX_WORKERS = 10


def github_api(endpoint: str, token: str | None = None) -> Any:
    """Make a GitHub REST API request and return parsed JSON."""
    url = f"https://api.github.com{endpoint}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "likelihoodlum",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as exc:
        if exc.code == 403:
            print(
                "⚠  GitHub API rate limit hit. Provide a --token to increase limits.",
                file=sys.stderr,
            )
        elif exc.code == 404:
            print(f"⚠  Repository not found: {endpoint}", file=sys.stderr)
        raise SystemExit(1)


def fetch_repo_metadata(owner: str, repo: str, token: str | None) -> dict:
    """Fetch repository metadata (creation date, description, etc.)."""
    return github_api(f"/repos/{owner}/{repo}", token)


def fetch_commits(
    owner: str, repo: str, token: str | None, branch: str | None, max_commits: int
) -> list[dict]:
    """Fetch up to *max_commits* commits from the repo."""
    commits: list[dict] = []
    page = 1
    per_page = min(max_commits, 100)
    while len(commits) < max_commits:
        endpoint = f"/repos/{owner}/{repo}/commits?per_page={per_page}&page={page}"
        if branch:
            endpoint += f"&sha={branch}"
        batch = github_api(endpoint, token)
        if not batch:
            break
        commits.extend(batch)
        if len(batch) < per_page:
            break
        page += 1
    return commits[:max_commits]


def fetch_commit_detail(owner: str, repo: str, sha: str, token: str | None) -> dict:
    """Fetch full detail (including stats and file list) for a single commit."""
    return github_api(f"/repos/{owner}/{repo}/commits/{sha}", token)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def parse_iso(datestr: str) -> datetime:
    """Parse a GitHub ISO-8601 date string into a timezone-aware datetime."""
    datestr = datestr.replace("Z", "+00:00")
    return datetime.fromisoformat(datestr)


def compute_authored_changes(files: list[dict]) -> dict:
    """
    Separate a commit's file changes into authored vs generated.

    Returns dict with:
        authored_additions, authored_deletions, authored_total,
        generated_total, authored_files, generated_files
    """
    authored_add = 0
    authored_del = 0
    generated_total = 0
    authored_filenames = []
    generated_filenames = []

    for f in files:
        filename = f.get("filename", "")
        changes = f.get("additions", 0) + f.get("deletions", 0)
        if is_generated_file(filename):
            generated_total += changes
            generated_filenames.append(filename)
        else:
            authored_add += f.get("additions", 0)
            authored_del += f.get("deletions", 0)
            authored_filenames.append(filename)

    return {
        "authored_additions": authored_add,
        "authored_deletions": authored_del,
        "authored_total": authored_add + authored_del,
        "generated_total": generated_total,
        "authored_files": authored_filenames,
        "generated_files": generated_filenames,
    }


def is_bot_author(name: str) -> bool:
    """Return True if the author name looks like a bot account."""
    lower = name.lower()
    for suffix in BOT_AUTHOR_SUFFIXES:
        if lower.endswith(suffix):
            return True
    return False


def compute_velocity(commits_chrono: list[dict]) -> list[dict]:
    """
    Given commits in chronological order (oldest first), compute lines-per-minute
    between consecutive commits by the same author.

    Uses authored_total (excluding generated files) for the calculation.
    Bot authors are excluded.
    """
    velocities: list[dict] = []
    prev_by_author: dict[str, dict] = {}

    for c in commits_chrono:
        author = c.get("author_login") or c.get("author_name") or "unknown"
        if is_bot_author(author):
            continue
        ts = c["timestamp"]
        authored_changes = c["authored_total"]

        if author in prev_by_author:
            prev = prev_by_author[author]
            gap = (ts - prev["timestamp"]).total_seconds() / 60.0
            if gap >= MIN_COMMIT_GAP_MINUTES:
                lpm = authored_changes / gap if gap > 0 else 0
                velocities.append(
                    {
                        "sha_from": prev["sha"][:8],
                        "sha_to": c["sha"][:8],
                        "author": author,
                        "gap_minutes": round(gap, 1),
                        "lines_changed": authored_changes,
                        "lines_per_minute": round(lpm, 2),
                    }
                )

        prev_by_author[author] = c

    return velocities


def build_sessions(commits_chrono: list[dict]) -> list[list[dict]]:
    """Group commits into coding sessions per author (excluding bots)."""
    by_author: dict[str, list[dict]] = defaultdict(list)
    for c in commits_chrono:
        author = c.get("author_login") or c.get("author_name") or "unknown"
        if is_bot_author(author):
            continue
        by_author[author].append(c)

    sessions: list[list[dict]] = []
    for author, author_commits in by_author.items():
        session: list[dict] = [author_commits[0]]
        for c in author_commits[1:]:
            gap = (c["timestamp"] - session[-1]["timestamp"]).total_seconds() / 60.0
            if gap > SESSION_GAP_MINUTES:
                sessions.append(session)
                session = [c]
            else:
                session.append(c)
        if session:
            sessions.append(session)

    return sessions


def analyze_messages(commits: list[dict]) -> dict:
    """Score commit messages for LLM-likeness."""
    pattern_hits = 0
    total = len(commits)
    flagged: list[str] = []

    for c in commits:
        msg = c.get("message", "").strip().split("\n")[0]
        msg_lower = msg.lower()
        for pat in LLM_MESSAGE_PATTERNS:
            if re.search(pat, msg_lower):
                pattern_hits += 1
                flagged.append(msg)
                break

    return {
        "total": total,
        "pattern_hits": pattern_hits,
        "ratio": round(pattern_hits / total, 3) if total else 0,
        "sample_flagged": flagged[:10],
    }


def trimmed_mean(values: list[float], trim_pct: float = 0.1) -> float:
    """Compute a trimmed mean, removing the top and bottom trim_pct of values."""
    if not values:
        return 0.0
    n = len(values)
    k = int(n * trim_pct)
    if k == 0 or n < 5:
        return statistics.mean(values)
    sorted_vals = sorted(values)
    trimmed = sorted_vals[k : n - k]
    return statistics.mean(trimmed) if trimmed else statistics.mean(values)


# ---------------------------------------------------------------------------
# Comment density helpers
# ---------------------------------------------------------------------------

# Patterns that indicate a comment line (added lines from patches)
_COMMENT_PATTERNS = [
    re.compile(r"^\s*//"),  # C/C++/Java/JS/Go/Rust/Swift/Dart
    re.compile(r"^\s*#(?!!)"),  # Python/Ruby/Shell (but not shebangs)
    re.compile(r"^\s*\*"),  # Block comment continuation
    re.compile(r"^\s*/\*"),  # Block comment start
    re.compile(r"^\s*\*/"),  # Block comment end
    re.compile(r"^\s*<!--"),  # HTML comment
    re.compile(r"^\s*--"),  # SQL/Lua comment
    re.compile(r'^\s*"""'),  # Python docstring
    re.compile(r"^\s*'''"),  # Python docstring
]


def _is_comment_line(line: str) -> bool:
    """Check if a source line looks like a comment."""
    for pat in _COMMENT_PATTERNS:
        if pat.match(line):
            return True
    return False


def analyze_comment_density(commits: list[dict]) -> dict:
    """
    Analyze the ratio of comment lines to code lines in added content.

    Uses the patch data stored on each commit's files.
    Returns dict with comment_ratio, total_comment_lines, total_code_lines.
    """
    total_comment = 0
    total_code = 0

    for c in commits:
        for patch_text in c.get("patches", []):
            for line in patch_text.split("\n"):
                # Only look at added lines (start with +, but not +++ header)
                if line.startswith("+") and not line.startswith("+++"):
                    content = line[1:]  # strip the leading +
                    stripped = content.strip()
                    if not stripped:
                        continue  # skip blank lines
                    if _is_comment_line(stripped):
                        total_comment += 1
                    else:
                        total_code += 1

    total = total_comment + total_code
    return {
        "total_comment_lines": total_comment,
        "total_code_lines": total_code,
        "comment_ratio": round(total_comment / total, 3) if total > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Commit time-of-day helpers
# ---------------------------------------------------------------------------


def analyze_commit_times(commits: list[dict]) -> dict:
    """
    Analyze the distribution of commit timestamps by hour of day.

    Returns dict with hour distribution, off-hours commit count, and
    off-hours high-velocity count.
    """
    hour_counts: Counter = Counter()
    off_hours_commits = 0  # commits between midnight and 6am
    total_non_bot = 0

    for c in commits:
        author = c.get("author_login") or c.get("author_name") or "unknown"
        if is_bot_author(author):
            continue
        total_non_bot += 1
        ts = c["timestamp"]
        hour = ts.hour
        hour_counts[hour] += 1
        if 0 <= hour < 6:
            off_hours_commits += 1

    off_hours_pct = off_hours_commits / total_non_bot if total_non_bot > 0 else 0

    return {
        "total_commits": total_non_bot,
        "off_hours_commits": off_hours_commits,
        "off_hours_pct": round(off_hours_pct, 3),
        "hour_distribution": dict(sorted(hour_counts.items())),
    }


# ---------------------------------------------------------------------------
# Diff entropy helpers
# ---------------------------------------------------------------------------


def _shannon_entropy(text: str) -> float:
    """Compute Shannon entropy of a string (bits per character)."""
    if not text:
        return 0.0
    freq: Counter = Counter(text)
    length = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def analyze_diff_entropy(commits: list[dict]) -> dict:
    """
    Analyze the Shannon entropy of diff content across commits.

    Lower entropy = more repetitive/predictable (LLM signal).
    Higher entropy = more varied/chaotic (human signal).

    Returns dict with median entropy, mean entropy, and per-commit entropies.
    """
    commit_entropies: list[float] = []

    for c in commits:
        # Concatenate all added lines from patches
        added_lines = []
        for patch_text in c.get("patches", []):
            for line in patch_text.split("\n"):
                if line.startswith("+") and not line.startswith("+++"):
                    added_lines.append(line[1:])

        if len(added_lines) < 5:
            continue  # skip tiny commits — not enough signal

        combined = "\n".join(added_lines)
        ent = _shannon_entropy(combined)
        commit_entropies.append(ent)

    if not commit_entropies:
        return {
            "median_entropy": None,
            "mean_entropy": None,
            "num_commits_measured": 0,
        }

    return {
        "median_entropy": round(statistics.median(commit_entropies), 3),
        "mean_entropy": round(statistics.mean(commit_entropies), 3),
        "num_commits_measured": len(commit_entropies),
    }


def score_repo(
    velocities: list[dict],
    sessions: list[list[dict]],
    msg_analysis: dict,
    commits: list[dict],
    repo_created_at: datetime | None = None,
) -> dict:
    """
    Compute a composite LLM-likelihood score from 0-100.

    Includes both positive signals (suspicious patterns) and negative
    signals (clearly human patterns) that actively reduce the score.
    """
    score = 0.0
    reasons: list[str] = []

    # --- 1. Velocity score (0-35 pts, can subtract up to -10) ---
    if velocities:
        lpms = [v["lines_per_minute"] for v in velocities]
        median_lpm = statistics.median(lpms)
        # Use trimmed mean to resist outlier skew
        tmean_lpm = trimmed_mean(lpms)

        suspicious_pct = sum(1 for l in lpms if l >= LPM_SUSPICIOUS) / len(lpms)
        very_suspicious_pct = sum(1 for l in lpms if l >= LPM_VERY_SUSPICIOUS) / len(
            lpms
        )

        # Positive signals
        if median_lpm >= LPM_VERY_SUSPICIOUS:
            score += 35
            reasons.append(
                f"Median velocity is extremely high ({median_lpm:.1f} lines/min "
                f"≈ {median_lpm * 60:.0f} lines/hr)"
            )
        elif median_lpm >= LPM_SUSPICIOUS:
            # Boost if trimmed mean is much higher than median (long tail of
            # fast intervals — classic LLM pattern)
            base = 22
            if tmean_lpm >= LPM_VERY_SUSPICIOUS:
                base = 30
            elif tmean_lpm >= median_lpm * 1.5:
                base = 26
            score += base
            reasons.append(
                f"Median velocity is suspiciously high ({median_lpm:.1f} lines/min "
                f"≈ {median_lpm * 60:.0f} lines/hr)"
            )
            if tmean_lpm >= median_lpm * 1.5:
                reasons.append(
                    f"Trimmed mean ({tmean_lpm:.1f} l/min) is {tmean_lpm / median_lpm:.1f}× "
                    f"the median — heavy tail of fast intervals"
                )
        elif median_lpm >= LPM_HUMAN_UPPER:
            score += 10
            reasons.append(
                f"Median velocity is above typical ({median_lpm:.1f} lines/min "
                f"≈ {median_lpm * 60:.0f} lines/hr)"
            )
        # Negative signals — clearly human pace
        elif median_lpm < LPM_CLEARLY_HUMAN:
            penalty = -10
            score += penalty
            reasons.append(
                f"Median velocity is consistent with human coding "
                f"({median_lpm:.1f} lines/min ≈ {median_lpm * 60:.0f} lines/hr) [{penalty:+.0f}]"
            )

        # Suspicious interval percentage — use the broader suspicious_pct as
        # a secondary check alongside the very_suspicious_pct.
        if very_suspicious_pct > 0.4:
            pts = min(10, very_suspicious_pct * 15)
            score += pts
            reasons.append(
                f"{very_suspicious_pct * 100:.0f}% of intervals show very high velocity"
            )
        elif very_suspicious_pct > 0.2:
            pts = min(5, very_suspicious_pct * 10)
            score += pts
            reasons.append(
                f"{very_suspicious_pct * 100:.0f}% of intervals show high velocity"
            )
        elif suspicious_pct > 0.4:
            pts = min(5, suspicious_pct * 8)
            score += pts
            reasons.append(
                f"{suspicious_pct * 100:.0f}% of intervals exceed suspicious threshold"
            )

    else:
        reasons.append("Not enough commit pairs to measure velocity")

    # --- 2. Session productivity (0-20 pts, can subtract up to -5) ---
    session_productivities: list[float] = []
    for sess in sessions:
        if len(sess) < 2:
            continue
        duration = (sess[-1]["timestamp"] - sess[0]["timestamp"]).total_seconds() / 60.0
        # Use authored_total to avoid counting generated files
        total_lines = sum(c["authored_total"] for c in sess)
        if duration >= 5:
            session_productivities.append(total_lines / duration)

    if session_productivities:
        # Use trimmed mean to resist outlier sessions
        tmean_session_prod = trimmed_mean(session_productivities)
        median_session_prod = statistics.median(session_productivities)

        if median_session_prod >= LPM_VERY_SUSPICIOUS:
            score += 20
            reasons.append(
                f"Median session productivity is extreme "
                f"({median_session_prod:.1f} lines/min)"
            )
        elif median_session_prod >= LPM_SUSPICIOUS:
            score += 12
            reasons.append(
                f"Median session productivity is high "
                f"({median_session_prod:.1f} lines/min)"
            )
        elif median_session_prod >= LPM_HUMAN_UPPER:
            score += 5
            reasons.append(
                f"Session productivity is above average "
                f"({median_session_prod:.1f} lines/min)"
            )
        elif median_session_prod < LPM_CLEARLY_HUMAN:
            penalty = -5
            score += penalty
            reasons.append(
                f"Session productivity is consistent with human pace "
                f"({median_session_prod:.1f} lines/min) [{penalty:+.0f}]"
            )

    # --- 3. Commit size uniformity (0-15 pts) ---
    # Use authored_total so generated files don't affect this
    sizes = [c["authored_total"] for c in commits if c["authored_total"] > 0]
    if len(sizes) >= 5:
        mean_size = statistics.mean(sizes)
        stdev_size = statistics.stdev(sizes)
        cv = stdev_size / mean_size if mean_size > 0 else 0

        # Low coefficient of variation with large commits = suspicious
        # Humans are messy — their commits vary a LOT
        if cv < 0.3 and mean_size > 150:
            score += 15
            reasons.append(
                f"Commits are suspiciously uniform and large "
                f"(mean={mean_size:.0f} lines, CV={cv:.2f})"
            )
        elif cv < 0.4 and mean_size > 100:
            score += 8
            reasons.append(
                f"Commits are somewhat uniform in size "
                f"(mean={mean_size:.0f} lines, CV={cv:.2f})"
            )
        # High variation = human signal
        elif cv > 1.5:
            penalty = -5
            score += penalty
            reasons.append(
                f"Commit sizes vary widely — typical of human work "
                f"(CV={cv:.2f}) [{penalty:+.0f}]"
            )

    # --- 4. Commit message patterns (0-15 pts) ---
    msg_ratio = msg_analysis["ratio"]
    if msg_ratio > 0.6:
        score += 15
        reasons.append(
            f"{msg_ratio * 100:.0f}% of commit messages match LLM-typical patterns"
        )
    elif msg_ratio > 0.35:
        score += 8
        reasons.append(
            f"{msg_ratio * 100:.0f}% of commit messages match LLM-typical patterns"
        )
    elif msg_ratio > 0.15:
        score += 3
        reasons.append(
            f"{msg_ratio * 100:.0f}% of commit messages match LLM-typical patterns"
        )
    # Low message match but very high velocity = still suspicious
    elif msg_ratio <= 0.15 and velocities:
        lpms_check = [v["lines_per_minute"] for v in velocities]
        if statistics.median(lpms_check) >= LPM_SUSPICIOUS:
            score += 2
            reasons.append(
                f"Commit messages look clean but velocity is high — "
                f"possible curated LLM workflow"
            )

    # --- 5. Burst detection (0-15 pts) ---
    # A burst = commits with lots of authored code in a short window,
    # OR a session with extreme per-commit velocity.
    burst_count = 0
    high_velocity_session_count = 0
    for sess in sessions:
        if len(sess) >= 2:
            duration = (
                sess[-1]["timestamp"] - sess[0]["timestamp"]
            ).total_seconds() / 60.0
            total_authored = sum(c["authored_total"] for c in sess)
            if duration < 30 and total_authored > 300:
                burst_count += 1
            # Also flag longer sessions with extreme throughput
            elif duration >= 5 and total_authored / duration >= LPM_VERY_SUSPICIOUS:
                high_velocity_session_count += 1

    total_burst_signals = burst_count + high_velocity_session_count

    if total_burst_signals >= 5:
        score += 15
        reasons.append(
            f"{total_burst_signals} burst/high-velocity sessions detected "
            f"({burst_count} rapid bursts, {high_velocity_session_count} sustained high-velocity)"
        )
    elif total_burst_signals >= 3:
        score += 10
        reasons.append(f"{total_burst_signals} burst/high-velocity sessions detected")
    elif total_burst_signals >= 1:
        score += 4
        reasons.append(f"{total_burst_signals} burst/high-velocity session detected")

    # --- 6. Multi-author discount (0 to -10 pts) ---
    # Exclude bot accounts from the author count
    authors = set()
    for c in commits:
        a = c.get("author_login") or c.get("author_name") or "unknown"
        if not is_bot_author(a):
            authors.add(a)

    if len(authors) >= 5:
        penalty = -10
        score += penalty
        reasons.append(
            f"{len(authors)} distinct authors — multi-contributor project [{penalty:+.0f}]"
        )
    elif len(authors) >= 3:
        penalty = -5
        score += penalty
        reasons.append(f"{len(authors)} distinct authors [{penalty:+.0f}]")
    elif len(authors) == 1:
        score += 5
        reasons.append("Solo author — consistent with LLM-assisted workflow [+5]")

    # --- 7. High per-commit velocity (0-10 pts) ---
    # If individual commits are huge relative to the time gap, that's suspicious
    # even if session-level metrics are diluted.
    if velocities:
        lpms_all = [v["lines_per_minute"] for v in velocities]
        # Count intervals where velocity exceeds 50 lines/min (3000 lines/hr)
        extreme_count = sum(1 for l in lpms_all if l >= 50.0)
        extreme_pct = extreme_count / len(lpms_all)
        if extreme_pct >= 0.05:
            pts = min(10, round(extreme_pct * 30, 1))
            score += pts
            reasons.append(
                f"{extreme_count} commit intervals ({extreme_pct * 100:.0f}%) "
                f"show extreme velocity (>50 lines/min ≈ 3000 lines/hr) [{pts:+.0f}]"
            )

    # --- 8. Commit time-of-day analysis (0-5 pts) ---
    # High-velocity work during off-hours (midnight–6am) is suspicious.
    time_analysis = analyze_commit_times(commits)
    if time_analysis["total_commits"] >= 10:
        off_pct = time_analysis["off_hours_pct"]
        if off_pct > 0.3 and velocities:
            median_check = statistics.median(
                [v["lines_per_minute"] for v in velocities]
            )
            if median_check >= LPM_SUSPICIOUS:
                score += 5
                reasons.append(
                    f"{off_pct * 100:.0f}% of commits are off-hours (midnight–6am) "
                    f"with high velocity [+5]"
                )

    # --- 9. Comment density analysis (0-5 pts, can subtract up to -3) ---
    comment_analysis = analyze_comment_density(commits)
    comment_ratio = comment_analysis["comment_ratio"]
    if comment_analysis["total_code_lines"] >= 100:
        if comment_ratio >= 0.35:
            score += 5
            reasons.append(
                f"Comment density is very high ({comment_ratio * 100:.0f}% of added lines "
                f"are comments) — LLMs over-explain [+5]"
            )
        elif comment_ratio >= 0.25:
            score += 3
            reasons.append(
                f"Comment density is above average ({comment_ratio * 100:.0f}% of added lines "
                f"are comments) [+3]"
            )
        elif comment_ratio < 0.05 and comment_analysis["total_code_lines"] >= 500:
            penalty = -3
            score += penalty
            reasons.append(
                f"Very low comment density ({comment_ratio * 100:.1f}%) — "
                f"typical human laziness [{penalty:+.0f}]"
            )

    # --- 10. Diff entropy analysis (0-5 pts, can subtract up to -3) ---
    entropy_analysis = analyze_diff_entropy(commits)
    if (
        entropy_analysis["median_entropy"] is not None
        and entropy_analysis["num_commits_measured"] >= 10
    ):
        median_ent = entropy_analysis["median_entropy"]
        # Source code typically has entropy around 4.5-5.5 bits/char.
        # LLM code tends toward lower entropy (more formulaic/repetitive).
        # Very high entropy suggests varied, chaotic human work.
        if median_ent < 4.0:
            score += 5
            reasons.append(
                f"Diff entropy is low ({median_ent:.2f} bits/char) — "
                f"diffs are repetitive/formulaic [+5]"
            )
        elif median_ent < 4.3:
            score += 3
            reasons.append(
                f"Diff entropy is below average ({median_ent:.2f} bits/char) [+3]"
            )
        elif median_ent > 5.5:
            penalty = -3
            score += penalty
            reasons.append(
                f"Diff entropy is high ({median_ent:.2f} bits/char) — "
                f"varied, chaotic human work [{penalty:+.0f}]"
            )

    # --- 11. Project-scale plausibility (0-20 pts, can subtract up to -5) ---
    # Compare total authored output against the project's true lifespan.
    # Use the repo creation date (from GitHub metadata) when available,
    # otherwise fall back to the oldest fetched commit.
    if len(commits) >= 5:
        total_authored = sum(c["authored_total"] for c in commits)
        last_ts = commits[-1]["timestamp"]

        # Prefer the true repo creation date over the oldest fetched commit
        if repo_created_at is not None:
            first_ts = repo_created_at
        else:
            first_ts = commits[0]["timestamp"]

        calendar_days = max((last_ts - first_ts).total_seconds() / 86400, 1)

        # Count distinct active days (days with at least one non-bot commit)
        active_dates: set[str] = set()
        for c in commits:
            a = c.get("author_login") or c.get("author_name") or "unknown"
            if not is_bot_author(a):
                active_dates.add(c["timestamp"].strftime("%Y-%m-%d"))
        active_days = max(len(active_dates), 1)

        daily_output_calendar = total_authored / calendar_days
        daily_output_active = total_authored / active_days

        # Use active-day rate as the primary metric (fairer for weekend gaps)
        daily_rate = daily_output_active

        if daily_rate >= DAILY_OUTPUT_VERY_SUSPICIOUS:
            score += 20
            reasons.append(
                f"Project-scale output is implausible: {total_authored:,} authored lines "
                f"over {calendar_days:.0f} days ({active_days} active) "
                f"= {daily_rate:,.0f} lines/active day [+20]"
            )
        elif daily_rate >= DAILY_OUTPUT_SUSPICIOUS:
            score += 12
            reasons.append(
                f"Project-scale output is very high: {total_authored:,} authored lines "
                f"over {calendar_days:.0f} days ({active_days} active) "
                f"= {daily_rate:,.0f} lines/active day [+12]"
            )
        elif daily_rate >= DAILY_OUTPUT_HUMAN_UPPER:
            score += 5
            reasons.append(
                f"Project-scale output is above average: {total_authored:,} authored lines "
                f"over {calendar_days:.0f} days ({active_days} active) "
                f"= {daily_rate:,.0f} lines/active day [+5]"
            )
        elif daily_rate < DAILY_OUTPUT_CLEARLY_HUMAN and calendar_days >= 14:
            penalty = -5
            score += penalty
            reasons.append(
                f"Project-scale output is consistent with human pace: "
                f"{daily_rate:,.0f} lines/active day over {calendar_days:.0f} days [{penalty:+.0f}]"
            )

    # --- 12. Generated file ratio signal ---
    total_all_changes = sum(c["total_changes"] for c in commits)
    total_generated = sum(c["generated_total"] for c in commits)
    if total_all_changes > 0:
        gen_ratio = total_generated / total_all_changes
        if gen_ratio > 0.5:
            reasons.append(
                f"ℹ  {gen_ratio * 100:.0f}% of all line changes are in generated/vendor files "
                f"(excluded from velocity calculations)"
            )

    # Clamp to 0-100
    score = max(0, min(100, score))

    return {
        "score": round(score, 1),
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def verdict(score: float) -> str:
    if score >= 75:
        return "🤖 Very likely LLM-generated"
    elif score >= 50:
        return "🤖 Likely LLM-assisted"
    elif score >= 30:
        return "🤔 Possibly LLM-assisted"
    elif score >= 15:
        return "👤 Likely human-written"
    else:
        return "👤 Almost certainly human-written"


def score_bar(score: float, width: int = 30) -> str:
    """Render a visual score bar."""
    filled = int(score / 100 * width)
    empty = width - filled
    if score >= 50:
        char = "█"
    elif score >= 30:
        char = "▓"
    else:
        char = "░"
    return f"[{char * filled}{'·' * empty}] {score:.0f}/100"


def print_report(
    owner: str,
    repo: str,
    commits: list[dict],
    velocities: list[dict],
    sessions: list[list[dict]],
    msg_analysis: dict,
    result: dict,
    repo_created_at: datetime | None = None,
) -> None:
    w = 60
    print()
    print("=" * w)
    print(f"  LLM Code Detector Report")
    print(f"  Repository: {owner}/{repo}")
    print("=" * w)

    print(f"\n📊 Commits analyzed: {len(commits)}")
    if commits:
        span = commits[-1]["timestamp"] - commits[0]["timestamp"]
        print(f"📅 Time span: {span.days} days")

    # Author breakdown
    authors: dict[str, int] = defaultdict(int)
    for c in commits:
        a = c.get("author_login") or c.get("author_name") or "unknown"
        authors[a] += 1
    print(f"👥 Authors: {len(authors)}")
    for a, count in sorted(authors.items(), key=lambda x: -x[1])[:5]:
        print(f"   • {a}: {count} commits")

    # Generated file stats
    total_all = sum(c["total_changes"] for c in commits)
    total_authored = sum(c["authored_total"] for c in commits)
    total_generated = sum(c["generated_total"] for c in commits)
    print(f"\n📁 Line changes breakdown:")
    print(f"   Total:     {total_all:,}")
    print(f"   Authored:  {total_authored:,} (used for analysis)")
    print(f"   Generated: {total_generated:,} (filtered out)")

    # Project-scale output
    if commits:
        if repo_created_at is not None:
            first_ts = repo_created_at
        else:
            first_ts = commits[0]["timestamp"]
        last_ts = commits[-1]["timestamp"]
        calendar_days = max((last_ts - first_ts).total_seconds() / 86400, 1)
        active_dates = set()
        for c in commits:
            a = c.get("author_login") or c.get("author_name") or "unknown"
            if not is_bot_author(a):
                active_dates.add(c["timestamp"].strftime("%Y-%m-%d"))
        active_days = max(len(active_dates), 1)
        daily_rate = total_authored / active_days
        daily_rate_cal = total_authored / calendar_days
        print(f"\n📈 Project-scale output:")
        if repo_created_at is not None:
            print(f"   Repo created:         {repo_created_at.strftime('%Y-%m-%d')}")
        print(
            f"   Active days:          {active_days} (of {calendar_days:.0f} calendar days)"
        )
        print(f"   Lines/active day:     {daily_rate:,.0f}")
        print(f"   Lines/calendar day:   {daily_rate_cal:,.0f}")
        flag = ""
        if daily_rate >= DAILY_OUTPUT_VERY_SUSPICIOUS:
            flag = " ⚠️  (>5,000 — implausible for a human)"
        elif daily_rate >= DAILY_OUTPUT_SUSPICIOUS:
            flag = " ⚠️  (>2,000 — very high)"
        elif daily_rate >= DAILY_OUTPUT_HUMAN_UPPER:
            flag = " (above average)"
        if flag:
            print(f"   {flag}")

    # Time-of-day analysis
    time_analysis = analyze_commit_times(commits)
    if time_analysis["total_commits"] >= 10:
        print(f"\n🕐 Commit time distribution:")
        off_pct = time_analysis["off_hours_pct"]
        print(
            f"   Off-hours (midnight–6am): {time_analysis['off_hours_commits']}/{time_analysis['total_commits']} ({off_pct * 100:.0f}%)"
        )

    # Velocity stats
    if velocities:
        lpms = [v["lines_per_minute"] for v in velocities]
        median_lpm = statistics.median(lpms)
        tmean_lpm = trimmed_mean(lpms)
        print(f"\n⚡ Velocity (authored lines/min between commits):")
        print(f"   Median:        {median_lpm:.2f}  (≈ {median_lpm * 60:.0f} lines/hr)")
        print(f"   Trimmed mean:  {tmean_lpm:.2f}  (≈ {tmean_lpm * 60:.0f} lines/hr)")
        print(f"   Max:           {max(lpms):.2f}")
        suspicious_count = sum(1 for l in lpms if l >= LPM_SUSPICIOUS)
        print(
            f"   Intervals above suspicious threshold: {suspicious_count}/{len(lpms)}"
        )

        # Top 5 fastest intervals
        top = sorted(velocities, key=lambda v: -v["lines_per_minute"])[:5]
        print(f"\n🔥 Fastest commit intervals:")
        for v in top:
            flag = " ⚠️" if v["lines_per_minute"] >= LPM_SUSPICIOUS else ""
            print(
                f"   {v['sha_from']}→{v['sha_to']}  {v['lines_changed']} lines "
                f"in {v['gap_minutes']} min = {v['lines_per_minute']} l/min{flag}"
            )

    # Sessions
    multi_sessions = [s for s in sessions if len(s) >= 2]
    print(
        f"\n📋 Coding sessions (gap > {SESSION_GAP_MINUTES} min): {len(multi_sessions)}"
    )

    # Comment density
    comment_analysis = analyze_comment_density(commits)
    if comment_analysis["total_code_lines"] >= 100:
        print(
            f"\n💬 Comment density: {comment_analysis['comment_ratio'] * 100:.1f}% "
            f"({comment_analysis['total_comment_lines']:,} comment lines / "
            f"{comment_analysis['total_code_lines']:,} code lines)"
        )

    # Diff entropy
    entropy_analysis = analyze_diff_entropy(commits)
    if entropy_analysis["median_entropy"] is not None:
        print(
            f"\n🔀 Diff entropy: {entropy_analysis['median_entropy']:.2f} bits/char "
            f"(median across {entropy_analysis['num_commits_measured']} commits)"
        )

    # Messages
    print(
        f"\n💬 Commit messages matching LLM patterns: "
        f"{msg_analysis['pattern_hits']}/{msg_analysis['total']} "
        f"({msg_analysis['ratio'] * 100:.1f}%)"
    )
    if msg_analysis["sample_flagged"]:
        for m in msg_analysis["sample_flagged"][:5]:
            print(f'   • "{m}"')

    # Final score
    print(f"\n{'─' * w}")
    print(f"  🎯 LLM Likelihood Score: {score_bar(result['score'])}")
    print(f"  {verdict(result['score'])}")
    print(f"{'─' * w}")

    if result["reasons"]:
        print(f"\n📝 Reasoning:")
        for r in result["reasons"]:
            print(f"   • {r}")

    print(f"\n⚠  Disclaimer: This is a heuristic analysis and NOT definitive proof.")
    print(f"   Fast coding can also indicate copy-paste, boilerplate generators,")
    print(f"   IDE scaffolding, or simply an experienced developer.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_repo_arg(repo_arg: str) -> tuple[str, str]:
    """Parse 'owner/repo' from a URL or shorthand."""
    # https://github.com/owner/repo or https://github.com/owner/repo.git
    m = re.match(r"(?:https?://)?github\.com/([^/]+)/([^/.]+)", repo_arg)
    if m:
        return m.group(1), m.group(2)
    # owner/repo shorthand
    m = re.match(r"^([^/]+)/([^/]+)$", repo_arg)
    if m:
        return m.group(1), m.group(2)
    print(f"Error: Cannot parse repo from '{repo_arg}'", file=sys.stderr)
    print(
        "Expected format: owner/repo or https://github.com/owner/repo", file=sys.stderr
    )
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect whether a GitHub repo's code was likely written by an LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("repo", help="GitHub repo as owner/repo or full URL")
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub personal access token (or set GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--branch", default=None, help="Branch to analyze (default: repo default)"
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        default=200,
        help="Maximum number of commits to fetch (default: 200)",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output", help="Output results as JSON"
    )
    args = parser.parse_args()

    owner, repo = parse_repo_arg(args.repo)

    if not args.token:
        print(
            "💡 Tip: Set GITHUB_TOKEN env var or use --token for higher API rate limits.\n",
            file=sys.stderr,
        )

    # Fetch repo metadata (creation date) — 1 API call
    print(f"🔍 Fetching repo metadata for {owner}/{repo}...", file=sys.stderr)
    repo_meta = fetch_repo_metadata(owner, repo, args.token)
    repo_created_at = parse_iso(repo_meta.get("created_at", "2000-01-01T00:00:00Z"))

    # Fetch commits
    print(f"🔍 Fetching commits for {owner}/{repo}...", file=sys.stderr)
    raw_commits = fetch_commits(owner, repo, args.token, args.branch, args.max_commits)

    if not raw_commits:
        print("No commits found.", file=sys.stderr)
        raise SystemExit(1)

    # Filter out bot commits before fetching details — saves API calls
    human_commits = []
    bot_commits = []
    for rc in raw_commits:
        login = (rc.get("author") or {}).get("login", "")
        name = (rc.get("commit", {}).get("author") or {}).get("name", "")
        author_id = login or name or "unknown"
        if is_bot_author(author_id):
            bot_commits.append(rc)
        else:
            human_commits.append(rc)

    if bot_commits:
        print(
            f"   Found {len(raw_commits)} commits "
            f"({len(bot_commits)} bot commits skipped). "
            f"Fetching details for {len(human_commits)}...",
            file=sys.stderr,
        )
    else:
        print(
            f"   Found {len(raw_commits)} commits. Fetching details...",
            file=sys.stderr,
        )

    # Fetch detailed stats concurrently — dramatically faster than sequential
    def _fetch_one(idx_rc: tuple[int, dict]) -> tuple[int, dict, dict]:
        idx, rc = idx_rc
        sha = rc["sha"]
        detail = fetch_commit_detail(owner, repo, sha, args.token)
        return idx, rc, detail

    commits: list[dict] = []
    workers = min(MAX_WORKERS, len(human_commits)) or 1
    done_count = 0
    total_to_fetch = len(human_commits)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_one, (i, rc)): i for i, rc in enumerate(human_commits)
        }
        for future in as_completed(futures):
            done_count += 1
            if done_count % 20 == 0 or done_count == 1 or done_count == total_to_fetch:
                print(
                    f"   Processing commit {done_count}/{total_to_fetch}...",
                    file=sys.stderr,
                )

            idx, rc, detail = future.result()
            stats = detail.get("stats", {})
            files = detail.get("files", [])

            # Separate authored vs generated file changes
            change_breakdown = compute_authored_changes(files)

            # Extract patch text from authored files for comment/entropy analysis
            patches = []
            for f in files:
                fname = f.get("filename", "")
                if not is_generated_file(fname) and "patch" in f:
                    patches.append(f["patch"])

            commit_info = rc.get("commit", {})
            author_info = commit_info.get("author", {})
            committer_info = commit_info.get("committer", {})

            commits.append(
                {
                    "sha": rc["sha"],
                    "message": commit_info.get("message", ""),
                    "author_name": author_info.get("name", "unknown"),
                    "author_login": (rc.get("author") or {}).get("login"),
                    "timestamp": parse_iso(
                        author_info.get(
                            "date",
                            committer_info.get("date", "2000-01-01T00:00:00Z"),
                        )
                    ),
                    "additions": stats.get("additions", 0),
                    "deletions": stats.get("deletions", 0),
                    "total_changes": stats.get("total", 0),
                    "files_changed": len(files),
                    # Filtered metrics
                    "authored_total": change_breakdown["authored_total"],
                    "authored_additions": change_breakdown["authored_additions"],
                    "authored_deletions": change_breakdown["authored_deletions"],
                    "generated_total": change_breakdown["generated_total"],
                    # Patch content for comment/entropy analysis
                    "patches": patches,
                }
            )

    # Sort chronologically (oldest first)
    commits.sort(key=lambda c: c["timestamp"])

    # Run analyses
    velocities = compute_velocity(commits)
    sessions = build_sessions(commits)
    msg_analysis = analyze_messages(commits)
    result = score_repo(velocities, sessions, msg_analysis, commits, repo_created_at)

    # Pre-compute analyses used by both report and JSON output
    _comment_analysis = analyze_comment_density(commits)
    _entropy_analysis = analyze_diff_entropy(commits)
    _time_analysis = analyze_commit_times(commits)

    if args.json_output:
        velocity_lpms = [v["lines_per_minute"] for v in velocities]
        output = {
            "repository": f"{owner}/{repo}",
            "commits_analyzed": len(commits),
            "score": result["score"],
            "verdict": verdict(result["score"]),
            "reasons": result["reasons"],
            "velocity_stats": {
                "median_lpm": round(statistics.median(velocity_lpms), 2)
                if velocity_lpms
                else None,
                "trimmed_mean_lpm": round(trimmed_mean(velocity_lpms), 2)
                if velocity_lpms
                else None,
                "intervals": len(velocities),
            },
            "line_changes": {
                "total": sum(c["total_changes"] for c in commits),
                "authored": sum(c["authored_total"] for c in commits),
                "generated": sum(c["generated_total"] for c in commits),
            },
            "project_scale": (
                lambda: {
                    "repo_created_at": repo_created_at.isoformat()
                    if repo_created_at
                    else None,
                    "calendar_days": round(
                        max(
                            (
                                commits[-1]["timestamp"]
                                - (repo_created_at or commits[0]["timestamp"])
                            ).total_seconds()
                            / 86400,
                            1,
                        ),
                        1,
                    ),
                    "active_days": len(
                        set(
                            c["timestamp"].strftime("%Y-%m-%d")
                            for c in commits
                            if not is_bot_author(
                                c.get("author_login")
                                or c.get("author_name")
                                or "unknown"
                            )
                        )
                    ),
                    "lines_per_active_day": round(
                        sum(c["authored_total"] for c in commits)
                        / max(
                            len(
                                set(
                                    c["timestamp"].strftime("%Y-%m-%d")
                                    for c in commits
                                    if not is_bot_author(
                                        c.get("author_login")
                                        or c.get("author_name")
                                        or "unknown"
                                    )
                                )
                            ),
                            1,
                        ),
                        0,
                    ),
                }
            )()
            if commits
            else None,
            "comment_density": {
                "comment_ratio": _comment_analysis["comment_ratio"],
                "comment_lines": _comment_analysis["total_comment_lines"],
                "code_lines": _comment_analysis["total_code_lines"],
            },
            "diff_entropy": {
                "median": _entropy_analysis["median_entropy"],
                "mean": _entropy_analysis["mean_entropy"],
                "commits_measured": _entropy_analysis["num_commits_measured"],
            },
            "commit_times": {
                "off_hours_pct": _time_analysis["off_hours_pct"],
                "off_hours_commits": _time_analysis["off_hours_commits"],
            },
            "message_analysis": msg_analysis,
            "sessions": len(sessions),
            "authors": len(
                set(
                    c.get("author_login") or c.get("author_name") or "unknown"
                    for c in commits
                )
            ),
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_report(
            owner,
            repo,
            commits,
            velocities,
            sessions,
            msg_analysis,
            result,
            repo_created_at,
        )


if __name__ == "__main__":
    main()
