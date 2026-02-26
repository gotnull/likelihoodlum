#!/usr/bin/env python3
"""
LLM-Generated Code Detector

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
  5. Time-of-day patterns: perfectly regular commit timing can be suspicious.
  6. Commit message analysis: generic / overly-perfect commit messages.

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
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

try:
    from dotenv import load_dotenv
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

# Lines-per-minute thresholds
LPM_HUMAN_TYPICAL = 0.5  # ~30 LoC / hour – a productive human
LPM_SUSPICIOUS = 3.0  # ~180 LoC / hour – very fast
LPM_VERY_SUSPICIOUS = 8.0  # ~480 LoC / hour – almost certainly assisted

# Minimum minutes between commits to consider for velocity calc.
# Commits less than 1 min apart are likely amend/rebase artifacts.
MIN_COMMIT_GAP_MINUTES = 1.0

# Commit message patterns that are suspiciously "LLM-like"
LLM_MESSAGE_PATTERNS = [
    r"^(add|create|implement|update|fix|refactor|improve|enhance)\s",
    r"^(feat|fix|docs|style|refactor|perf|test|chore)\(.+\):\s",  # conventional commits (not bad per se, but scored lightly)
    r"initial commit",
    r"^update \S+$",
    r"^add \S+$",
]

# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------


def github_api(endpoint: str, token: str | None = None) -> Any:
    """Make a GitHub REST API request and return parsed JSON."""
    url = f"https://api.github.com{endpoint}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "llm-detector-script",
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
    """Fetch full detail (including stats) for a single commit."""
    return github_api(f"/repos/{owner}/{repo}/commits/{sha}", token)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def parse_iso(datestr: str) -> datetime:
    """Parse a GitHub ISO-8601 date string into a timezone-aware datetime."""
    # GitHub returns dates like 2024-01-15T08:30:00Z
    datestr = datestr.replace("Z", "+00:00")
    return datetime.fromisoformat(datestr)


def compute_velocity(commits_chrono: list[dict]) -> list[dict]:
    """
    Given commits in chronological order (oldest first), compute lines-per-minute
    between consecutive commits by the same author.

    Returns a list of dicts with velocity info.
    """
    velocities: list[dict] = []
    prev_by_author: dict[str, dict] = {}

    for c in commits_chrono:
        author = c.get("author_login") or c.get("author_name") or "unknown"
        ts = c["timestamp"]
        total_changes = c["total_changes"]

        if author in prev_by_author:
            prev = prev_by_author[author]
            gap = (ts - prev["timestamp"]).total_seconds() / 60.0
            if gap >= MIN_COMMIT_GAP_MINUTES:
                lpm = total_changes / gap if gap > 0 else 0
                velocities.append(
                    {
                        "sha_from": prev["sha"][:8],
                        "sha_to": c["sha"][:8],
                        "author": author,
                        "gap_minutes": round(gap, 1),
                        "lines_changed": total_changes,
                        "lines_per_minute": round(lpm, 2),
                    }
                )

        prev_by_author[author] = c

    return velocities


def build_sessions(commits_chrono: list[dict]) -> list[list[dict]]:
    """Group commits into coding sessions per author."""
    by_author: dict[str, list[dict]] = defaultdict(list)
    for c in commits_chrono:
        author = c.get("author_login") or c.get("author_name") or "unknown"
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
        msg = c.get("message", "").strip().split("\n")[0].lower()
        for pat in LLM_MESSAGE_PATTERNS:
            if re.search(pat, msg, re.IGNORECASE):
                pattern_hits += 1
                flagged.append(c.get("message", "").strip().split("\n")[0])
                break

    return {
        "total": total,
        "pattern_hits": pattern_hits,
        "ratio": round(pattern_hits / total, 3) if total else 0,
        "sample_flagged": flagged[:10],
    }


def score_repo(
    velocities: list[dict],
    sessions: list[list[dict]],
    msg_analysis: dict,
    commits: list[dict],
) -> dict:
    """
    Compute a composite LLM-likelihood score from 0-100.
    """
    score = 0.0
    reasons: list[str] = []

    # --- 1. Velocity score (0-35 pts) ---
    if velocities:
        lpms = [v["lines_per_minute"] for v in velocities]
        median_lpm = statistics.median(lpms)
        suspicious_pct = sum(1 for l in lpms if l >= LPM_SUSPICIOUS) / len(lpms)
        very_suspicious_pct = sum(1 for l in lpms if l >= LPM_VERY_SUSPICIOUS) / len(
            lpms
        )

        if median_lpm >= LPM_VERY_SUSPICIOUS:
            score += 35
            reasons.append(
                f"Median velocity is extremely high ({median_lpm:.1f} lines/min ≈ {median_lpm * 60:.0f} lines/hr)"
            )
        elif median_lpm >= LPM_SUSPICIOUS:
            score += 22
            reasons.append(
                f"Median velocity is suspiciously high ({median_lpm:.1f} lines/min ≈ {median_lpm * 60:.0f} lines/hr)"
            )
        elif median_lpm >= LPM_HUMAN_TYPICAL:
            score += 8
            reasons.append(
                f"Median velocity is above typical human rate ({median_lpm:.1f} lines/min)"
            )

        if very_suspicious_pct > 0.3:
            score += min(10, very_suspicious_pct * 15)
            reasons.append(
                f"{very_suspicious_pct * 100:.0f}% of commit intervals show very high velocity"
            )
    else:
        reasons.append("Not enough commit pairs to measure velocity")

    # --- 2. Session productivity (0-20 pts) ---
    session_productivities: list[float] = []
    for sess in sessions:
        if len(sess) < 2:
            continue
        duration = (sess[-1]["timestamp"] - sess[0]["timestamp"]).total_seconds() / 60.0
        total_lines = sum(c["total_changes"] for c in sess)
        if duration >= 5:
            session_productivities.append(total_lines / duration)

    if session_productivities:
        median_session_prod = statistics.median(session_productivities)
        if median_session_prod >= LPM_VERY_SUSPICIOUS:
            score += 20
            reasons.append(
                f"Median session productivity is extreme ({median_session_prod:.1f} lines/min)"
            )
        elif median_session_prod >= LPM_SUSPICIOUS:
            score += 12
            reasons.append(
                f"Median session productivity is high ({median_session_prod:.1f} lines/min)"
            )
        elif median_session_prod >= LPM_HUMAN_TYPICAL * 2:
            score += 5
            reasons.append(
                f"Session productivity is above average ({median_session_prod:.1f} lines/min)"
            )

    # --- 3. Commit size uniformity (0-15 pts) ---
    sizes = [c["total_changes"] for c in commits if c["total_changes"] > 0]
    if len(sizes) >= 5:
        mean_size = statistics.mean(sizes)
        stdev_size = statistics.stdev(sizes)
        cv = stdev_size / mean_size if mean_size > 0 else 0
        # Low coefficient of variation with large commits = suspicious
        if cv < 0.4 and mean_size > 100:
            score += 15
            reasons.append(
                f"Commits are uniformly large (mean={mean_size:.0f}, CV={cv:.2f})"
            )
        elif cv < 0.6 and mean_size > 80:
            score += 8
            reasons.append(
                f"Commits are somewhat uniform in size (mean={mean_size:.0f}, CV={cv:.2f})"
            )

    # --- 4. Commit message patterns (0-15 pts) ---
    msg_ratio = msg_analysis["ratio"]
    if msg_ratio > 0.7:
        score += 15
        reasons.append(
            f"{msg_ratio * 100:.0f}% of commit messages match LLM-typical patterns"
        )
    elif msg_ratio > 0.4:
        score += 8
        reasons.append(
            f"{msg_ratio * 100:.0f}% of commit messages match LLM-typical patterns"
        )
    elif msg_ratio > 0.2:
        score += 3
        reasons.append(
            f"{msg_ratio * 100:.0f}% of commit messages match LLM-typical patterns"
        )

    # --- 5. Burst detection (0-15 pts) ---
    burst_count = 0
    for sess in sessions:
        if len(sess) >= 3:
            duration = (
                sess[-1]["timestamp"] - sess[0]["timestamp"]
            ).total_seconds() / 60.0
            total_lines = sum(c["total_changes"] for c in sess)
            if duration < 30 and total_lines > 500:
                burst_count += 1

    if burst_count >= 5:
        score += 15
        reasons.append(f"{burst_count} burst sessions detected (>500 lines in <30 min)")
    elif burst_count >= 2:
        score += 8
        reasons.append(f"{burst_count} burst sessions detected")
    elif burst_count >= 1:
        score += 3
        reasons.append(f"{burst_count} burst session detected")

    score = min(score, 100)

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


def print_report(
    owner: str,
    repo: str,
    commits: list[dict],
    velocities: list[dict],
    sessions: list[list[dict]],
    msg_analysis: dict,
    result: dict,
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

    # Velocity stats
    if velocities:
        lpms = [v["lines_per_minute"] for v in velocities]
        print(f"\n⚡ Velocity (lines/min between commits):")
        print(
            f"   Median: {statistics.median(lpms):.2f}  (≈ {statistics.median(lpms) * 60:.0f} lines/hr)"
        )
        print(f"   Mean:   {statistics.mean(lpms):.2f}")
        print(f"   Max:    {max(lpms):.2f}")
        print(
            f"   Intervals above suspicious threshold: "
            f"{sum(1 for l in lpms if l >= LPM_SUSPICIOUS)}/{len(lpms)}"
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
        f"\n🕐 Coding sessions (gap > {SESSION_GAP_MINUTES} min): {len(multi_sessions)}"
    )

    # Messages
    print(
        f"\n💬 Commit messages matching LLM patterns: "
        f"{msg_analysis['pattern_hits']}/{msg_analysis['total']} ({msg_analysis['ratio'] * 100:.1f}%)"
    )
    if msg_analysis["sample_flagged"]:
        for m in msg_analysis["sample_flagged"][:5]:
            print(f'   • "{m}"')

    # Final score
    print(f"\n{'─' * w}")
    print(f"  🎯 LLM Likelihood Score: {result['score']}/100")
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

    # Fetch commits
    print(f"🔍 Fetching commits for {owner}/{repo}...", file=sys.stderr)
    raw_commits = fetch_commits(owner, repo, args.token, args.branch, args.max_commits)

    if not raw_commits:
        print("No commits found.", file=sys.stderr)
        raise SystemExit(1)

    print(f"   Found {len(raw_commits)} commits. Fetching details...", file=sys.stderr)

    # Fetch detailed stats for each commit (this is the slow part)
    commits: list[dict] = []
    for i, rc in enumerate(raw_commits):
        sha = rc["sha"]
        if (i + 1) % 20 == 0 or i == 0:
            print(
                f"   Processing commit {i + 1}/{len(raw_commits)}...", file=sys.stderr
            )

        detail = fetch_commit_detail(owner, repo, sha, args.token)
        stats = detail.get("stats", {})

        commit_info = rc.get("commit", {})
        author_info = commit_info.get("author", {})
        committer_info = commit_info.get("committer", {})

        commits.append(
            {
                "sha": sha,
                "message": commit_info.get("message", ""),
                "author_name": author_info.get("name", "unknown"),
                "author_login": (rc.get("author") or {}).get("login"),
                "timestamp": parse_iso(
                    author_info.get(
                        "date", committer_info.get("date", "2000-01-01T00:00:00Z")
                    )
                ),
                "additions": stats.get("additions", 0),
                "deletions": stats.get("deletions", 0),
                "total_changes": stats.get("total", 0),
                "files_changed": len(detail.get("files", [])),
            }
        )

    # Sort chronologically (oldest first)
    commits.sort(key=lambda c: c["timestamp"])

    # Run analyses
    velocities = compute_velocity(commits)
    sessions = build_sessions(commits)
    msg_analysis = analyze_messages(commits)
    result = score_repo(velocities, sessions, msg_analysis, commits)

    if args.json_output:
        output = {
            "repository": f"{owner}/{repo}",
            "commits_analyzed": len(commits),
            "score": result["score"],
            "verdict": verdict(result["score"]),
            "reasons": result["reasons"],
            "velocity_stats": {
                "median_lpm": round(
                    statistics.median([v["lines_per_minute"] for v in velocities]), 2
                )
                if velocities
                else None,
                "intervals": len(velocities),
            },
            "message_analysis": msg_analysis,
            "sessions": len(sessions),
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_report(owner, repo, commits, velocities, sessions, msg_analysis, result)


if __name__ == "__main__":
    main()
