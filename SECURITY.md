# Security Policy

## Reporting a Vulnerability

Found a security issue? **Don't open a public issue.**

Email **security@churnguard.dev** (or DM a maintainer on GitHub) with:
- What you found
- How to reproduce it
- Potential impact

We'll acknowledge within 48 hours and keep you updated.

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest release | ✅ Yes |
| Older releases | ❌ No |

Security fixes go to `main` and get released ASAP.

## What We Consider Security Issues

- Code execution via maliciously crafted input data (CSV, YAML config)
- Dependency vulnerabilities (supply chain)
- Path traversal in file operations
- SHAP/model serialization exploits

## What We Don't Consider Security Issues

- Bugs requiring local file access you already have
- Denial-of-service via resource exhaustion (large datasets)
- Issues in optional `viz` dependencies (matplotlib, seaborn)

## Disclosure Timeline

1. **Day 0**: Private report
2. **Day 1-2**: Triage + confirm
3. **Day 7-30**: Fix + test
4. **Day 30+**: Public disclosure, release, credit (unless you want anonymity)

## Security Practices

- Minimal dependencies — only well-maintained, popular packages
- `pip-audit` runs in CI on every PR
- No `eval()`, `exec()`, `pickle.load()` on untrusted input
- YAML config uses `yaml.safe_load()` only
- File operations use `pathlib` with validation

Run locally:
```bash
pip-audit
```