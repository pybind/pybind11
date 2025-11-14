#!/usr/bin/env -S uv run

# /// script
# dependencies = ["ghapi", "rich"]
# ///

from __future__ import annotations

import re

import ghapi.all
from rich import print
from rich.syntax import Syntax

MD_ENTRY = re.compile(
    r"""
    \#\#\ Suggested\ changelog\ entry:     # Match the heading exactly
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
print()


api = ghapi.all.GhApi(owner="pybind", repo="pybind11")

issues_pages = ghapi.page.paged(
    api.issues.list_for_repo, labels="needs changelog", state="closed"
)
issues = (issue for page in issues_pages for issue in page)
missing = []
old = []
cats_descr = {
    "feat": "New Features",
    "feat(types)": "",
    "feat(cmake)": "",
    "fix": "Bug fixes",
    "fix(types)": "",
    "fix(cmake)": "",
    "fix(free-threading)": "",
    "docs": "Documentation",
    "tests": "Tests",
    "ci": "CI",
    "chore": "Other",
    "chore(cmake)": "",
    "unknown": "Uncategorised",
}
cats: dict[str, list[str]] = {c: [] for c in cats_descr}

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
    if msg.startswith("* "):
        msg = msg[2:]
    if not msg.startswith("- "):
        msg = "- " + msg
    if not msg.endswith("."):
        msg += "."
    if msg == "- Placeholder.":
        missing.append(issue)
        continue

    msg += f"\n  [#{issue.number}]({issue.html_url})"
    for cat, cat_list in cats.items():
        if issue.title.lower().startswith(f"{cat}:"):
            cat_list.append(msg)
            break
    else:
        cats["unknown"].append(msg)

for cat, msgs in cats.items():
    if msgs:
        desc = cats_descr[cat]
        print(f"[bold]{desc}:" if desc else f"<!-- {cat} -->")
        print()
        for msg in msgs:
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
