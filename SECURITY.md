# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | ✅ Yes             |
| < 1.0   | ❌ No              |

## Reporting a Vulnerability

If you discover a security vulnerability in Likelihoodlum, please report it responsibly.

### What to report

- Vulnerabilities in how GitHub tokens are handled or stored
- Issues with the `.env` file parsing that could leak credentials
- Injection risks via repository names, commit messages, or API responses
- Any way the tool could be used to exfiltrate data

### How to report

**Do NOT open a public issue for security vulnerabilities.**

Instead, please email: **gotnull@users.noreply.github.com**

Include:
1. A description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if you have one)

### What to expect

- **Acknowledgment** within 48 hours
- **Assessment** within 7 days
- **Fix or mitigation** as quickly as possible, depending on severity
- **Credit** in the changelog and release notes (unless you prefer anonymity)

## Security Considerations

Likelihoodlum handles GitHub Personal Access Tokens. The tool follows these practices:

- Tokens are **never logged, printed, or written to disk** by the tool
- The `.env` file is **gitignored by default**
- Tokens passed via `--token` may appear in shell history — using `.env` or `GITHUB_TOKEN` env var is recommended
- All API requests use HTTPS
- No data is sent anywhere other than the GitHub API

## Scope

This policy covers the `likelihoodlum` Python tool and its repository. It does **not** cover:

- The GitHub API itself
- Third-party dependencies (e.g. `python-dotenv`, which is optional)
- Your own GitHub token's security — that's between you and GitHub