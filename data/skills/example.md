---
name: Python Best Practices
description: General Python coding conventions for this project
---
# Python Best Practices

- Use type hints on all function signatures
- Prefer `pathlib.Path` over `os.path` string manipulation
- Use `f-strings` for string formatting, not `.format()` or `%`
- Keep functions under 30 lines; extract helpers for complex logic
- Use `dataclasses` or `TypedDict` for structured data, not raw dicts
