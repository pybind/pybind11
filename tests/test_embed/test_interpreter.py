from __future__ import annotations

import sys

import pytest

if sys.platform.startswith("emscripten"):
    pytest.skip(
        "Test not implemented from single wheel on Pyodide", allow_module_level=True
    )

from widget_module import Widget


class DerivedWidget(Widget):
    def __init__(self, message):
        super().__init__(message)

    def the_answer(self):
        return 42

    def argv0(self):
        return sys.argv[0]
