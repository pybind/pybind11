#!/usr/bin/env -S uv run

# /// script
# dependencies = ["ghapi>=2", "rich"]
# ///

from __future__ import annotations

import os
import re
import subprocess

import ghapi.all
from rich import print
from rich.syntax import Syntax

MD_ENTRY = re.compile(
    r"""
    ^\#+\ Suggested\ changelog\ entry:?\s*$ # Match the heading, colon optional
    (?:\s*<!--.*?-->)?                     # Optionally match one comment
    (?P<content>.*?)                       # Lazily capture content until...
    (?=                                    # Lookahead for one of the following:
        ^-{3,}\s*$                         #   A line with 3 or more dashes
      | ^<!--\s*readthedocs                #   A comment starting with 'readthedocs'
      | ^\#\#                              #   A new heading
      | \Z                                 #   End of string
    )
    """,
    re.DOTALL | re.VERBOSE | re.MULTILINE,
)
# Conventional commit prefix, such as "fix(cmake)!:"
TITLE_CAT = re.compile(r"(?P<cat>\w+)(?:\((?P<sub>[^)]+)\))?!?:")

print()


def get_token() -> str | None:
    """
    Unauthenticated requests get a shared CDN cache ("Cache-Control: public"),
    which returns stale bodies and labels. A token makes the cache private.
    """
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    try:
        result = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        print("[yellow]No GitHub token found; results may be stale (cached).")
        return None
    return result.stdout.strip()


LABEL = "needs changelog"

api = ghapi.all.GhApi(owner="pybind", repo="pybind11", token=get_token(), sync=True)

issues_pages = ghapi.page.sync_paged(
    api.issues.list_for_repo, labels=LABEL, state="closed", per_page=100
)
# The server-side label filter lags behind label removals, so re-check each issue
issues = (
    issue
    for page in issues_pages
    for issue in page
    if any(label.name == LABEL for label in issue.labels)
)
missing = []
old = []
cats_descr = {
    "feat": "New Features",
    "fix": "Bug fixes",
    "docs": "Documentation",
    "tests": "Tests",
    "ci": "CI",
    "chore": "Other",
    "unknown": "Uncategorised",
}
# Each main category maps its subcategories ("" if there is none) to entries
cats: dict[str, dict[str, list[str]]] = {c: {} for c in cats_descr}

for issue in issues:
    if "```rst" in issue.body:
        old.append(issue)
        continue

    changelog = MD_ENTRY.search(issue.body or "")
    if not changelog:
        missing.append(issue)
        continue

    msg = changelog.group("content").strip()
    if not msg:
        missing.append(issue)
        continue
    msg = msg.removeprefix("* ")
    if not msg.startswith("- "):
        msg = "- " + msg
    if not msg.endswith("."):
        msg += "."
    if msg == "- Placeholder.":
        missing.append(issue)
        continue

    msg += f"\n  [#{issue.number}]({issue.html_url})"
    title = TITLE_CAT.match(issue.title)
    cat = title.group("cat").lower() if title else "unknown"
    sub = (title.group("sub") or "").lower() if title else ""
    if cat not in cats:
        cat, sub = "unknown", ""
    cats[cat].setdefault(sub, []).append(msg)

for cat, subs in cats.items():
    if subs:
        print(f"[bold]{cats_descr[cat]}:")
        print()
        # An empty subcategory sorts first, so plain entries lead the section
        for sub in sorted(subs):
            if sub:
                print(f"<!-- {cat}({sub}) -->")
                print()
            for msg in subs[sub]:
                print(Syntax(msg, "md", theme="ansi_light", word_wrap=True))
                print()
            print()

if missing:
    print()
    print("[blue]" + "-" * 30)
    print()

    for issue in missing:
        print(f"[red bold]Missing:[/red bold][red] {issue.title}")
        print(f"[red]  {issue.html_url}\n")

    print("[bold]Template:\n")
    msg = "## Suggested changelog entry:"
    print(Syntax(msg, "md", theme="ansi_light"))

if old:
    print()
    print("[red]" + "-" * 30)
    print()

    for issue in old:
        print(f"[red bold]Old:[/red bold][red] {issue.title}")
        print(f"[red]  {issue.html_url}\n")

print()
