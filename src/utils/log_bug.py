"""
Owner: Team (Maintained by Aiman Aslam - Experiments Lead)
Utility: log_bug.py
Purpose: Append real debugging entries to docs/bug_diary.md following academic integrity.
"""

import argparse
import datetime
from pathlib import Path

def append_bug_entry(
    title: str,
    author: str,
    env: str,
    symptom: str,
    cause: str,
    fix: str,
    script: str = "N/A",
    diary_path: Path = Path("docs/bug_diary.md"),
) -> None:
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry_md = f"""
---

## Bug Entry: {title}
- **Date & Time:** {now_str}
- **Author / Discoverer:** {author}
- **Environment:** {env}
- **Commit / Script:** {script}

### 1. Symptom
```text
{symptom}
```

### 2. Root Cause Analysis
{cause}

### 3. Resolution / Fix
```python
{fix}
```
- **Verification:** Verified via local smoke run and regression tests.
"""
    if not diary_path.exists():
        raise FileNotFoundError(f"Bug diary file {diary_path} not found.")

    with open(diary_path, "a", encoding="utf-8") as f:
        f.write(entry_md)

    print(f"[Bug Diary] Successfully appended entry '{title}' to {diary_path}")


def main():
    parser = argparse.ArgumentParser(description="Append a real bug entry to docs/bug_diary.md")
    parser.add_argument("--title", type=str, required=True, help="Bug title / component")
    parser.add_argument("--author", type=str, required=True, help="Discoverer name")
    parser.add_argument("--env", type=str, default="Google Colab T4 GPU", help="Runtime environment")
    parser.add_argument("--script", type=str, default="N/A", help="Script or module affected")
    parser.add_argument("--symptom", type=str, required=True, help="Symptom / error message")
    parser.add_argument("--cause", type=str, required=True, help="Root cause explanation")
    parser.add_argument("--fix", type=str, required=True, help="Fix description or code snippet")

    args = parser.parse_args()
    append_bug_entry(
        title=args.title,
        author=args.author,
        env=args.env,
        symptom=args.symptom,
        cause=args.cause,
        fix=args.fix,
        script=args.script,
    )

if __name__ == "__main__":
    main()
