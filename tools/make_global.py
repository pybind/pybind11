#!/usr/bin/env -S uv run -q

# /// script
# dependencies = ["tomlkit"]
# ///
from __future__ import annotations

from pathlib import Path

import tomlkit

DIR = Path(__file__).parent.resolve()
PYPROJECT = DIR.parent / "pyproject.toml"


def get_global() -> str:
    pyproject = tomlkit.parse(PYPROJECT.read_text())
    del pyproject["tool"]["scikit-build"]["generate"]
    del pyproject["project"]["entry-points"]
    del pyproject["project"]["scripts"]
    del pyproject["tool"]["scikit-build"]["metadata"]["optional-dependencies"]
    pyproject["project"]["name"] = "pybind11-global"
    pyproject["tool"]["scikit-build"]["experimental"] = True
    pyproject["tool"]["scikit-build"]["wheel"]["install-dir"] = "/data"
    pyproject["tool"]["scikit-build"]["wheel"]["packages"] = []

    result = tomlkit.dumps(pyproject)
    assert isinstance(result, str)
    return result


if __name__ == "__main__":
    print(get_global())
