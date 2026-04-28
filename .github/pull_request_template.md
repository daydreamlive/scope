## Description

<!-- What does this PR change, and why? -->

## Type of Change

- [ ] Bugfix
- [ ] Feature
- [ ] Refactor
- [ ] Docs
- [ ] CI / Infrastructure

## Regression Test (bugfix PRs only)

If this PR is a bugfix, please include a regression test in `product-tests/regression/`.
Without one, the bug can quietly regress later. See `CLAUDE.md` → "Regression Tests for Bugfix PRs".

```bash
# Recommended — auto-generate from your bug description:
/product-test-writer

# Or manually:
cp product-tests/_templates/regression.py.tpl \
   product-tests/regression/pr_<your_pr_number>_<slug>.py
```

- [ ] Added a regression test (or N/A — not a bugfix)
- [ ] Verified the test reds on the buggy commit and greens on the fix commit (or N/A)

## Manual Testing

<!-- How did you verify this works? Steps a reviewer can follow. -->

## Pre-flight Checklist

- [ ] Lint passes (`uv run ruff check src/` and `npm run lint` from `frontend/`)
- [ ] Frontend builds (`npm run build` from `frontend/`)
- [ ] Server starts cleanly (`uv run daydream-scope`)
- [ ] Commits are signed off (DCO: `git commit -s`)
