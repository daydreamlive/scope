## Description

<!-- Describe the changes in this PR -->

## Type of Change

- [ ] Bugfix (fixes an issue)
- [ ] Feature (adds functionality)
- [ ] Refactor
- [ ] Documentation
- [ ] CI/Infrastructure

## Testing

<!-- For bugfixes: Did you add a regression test? -->

### Regression Test

**If this is a bugfix**, please add a regression test to prevent this bug from recurring:

```bash
# Auto-generate from bug description (recommended):
/product-test-writer

# Or manually:
cp product-tests/_templates/regression.py.tpl \
   product-tests/regression/pr_<your_pr_number>_<slug>.py
```

See `CLAUDE.md` → "Regression Tests for Bugfix PRs" for details.

- [ ] Added regression test (if bugfix)
- [ ] Verified test reds before fix, greens after fix
- [ ] Test uses `@scenario` decorator and fits on one screen

### Manual Testing

<!-- Describe how you tested the changes -->

## Checklist

- [ ] Code follows style guidelines (`npm run lint:fix` / `ruff check --fix`)
- [ ] All commits are signed off (`git commit -s`)
- [ ] Changes don't break existing tests (`uv run pytest tests/`)
- [ ] Frontend builds (`npm run build`)
- [ ] Server starts without errors (`uv run daydream-scope`)
