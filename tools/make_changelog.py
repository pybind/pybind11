#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

import ghapi.core


ENTRY = re.compile(
    r"""
    Suggested \s changelog \s entry:
    .*
    ```rst
    \s*
    (.*?)
    \s*
    ```
""",
    re.DOTALL | re.VERBOSE,
)


api = ghapi.core.GhApi(owner="pybind", repo="pybind11")

issues = api.issues.list_for_repo(labels="needs changelog", state="closed")
missing = []

for issue in issues:
    changelog = ENTRY.findall(issue.body)
    if changelog:
        (msg,) = changelog
        if not msg.startswith("* "):
            msg = "* " + msg
        if not msg.endswith("."):
            msg += "."

        print(msg)
        print(f"  `#{issue.number} <{issue.html_url}>`_\n")

    else:
        missing.append(issue)

if missing:
    print()
    print("-" * 30)
    print()

    for issue in missing:
        print(f"Missing: {issue.title}")
        print(f"  {issue.html_url}")
