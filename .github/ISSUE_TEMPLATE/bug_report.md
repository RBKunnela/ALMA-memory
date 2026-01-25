---
name: Bug Report
about: Report a bug to help us improve ALMA
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description
A clear and concise description of the bug.

## To Reproduce
Steps to reproduce the behavior:
1. Initialize ALMA with '...'
2. Call method '...'
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Code Sample
```python
# Minimal code to reproduce the issue
from alma import ALMA

alma = ALMA.from_config(".alma/config.yaml")
# ...
```

## Error Message
```
Paste full error traceback here
```

## Environment
- ALMA version: (run `pip show alma-memory`)
- Python version: (run `python --version`)
- OS: (e.g., Windows 11, Ubuntu 22.04, macOS 14)
- Storage backend: (sqlite/azure/file)
- Embedding provider: (local/azure)

## Additional Context
Add any other context about the problem here.
