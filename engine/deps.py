"""Check installed Python packages against the pinned versions in requirements.txt."""
import os
import re
from importlib import metadata

REQ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "requirements.txt")


def check() -> dict:
    problems, ok = [], 0
    try:
        lines = open(REQ, encoding="utf-8").read().splitlines()
    except OSError:
        return {"ok": False, "problems": ["requirements.txt not found"]}
    for line in lines:
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", line.strip())
        if not m:
            continue
        name, want = m.groups()
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            problems.append(f"{name} is not installed (need {want}). Run: npm run setup")
            continue
        if have.split("+")[0] != want.split("+")[0]:
            problems.append(f"{name} is {have}, expected {want}. Run: npm run setup")
        else:
            ok += 1
    return {"ok": not problems, "checked": ok, "problems": problems}
