from __future__ import annotations

import os

import env


def test_enum_import_exit_no_crash():
    # Modeled after reproducer under issue #5976
    env.check_script_success_in_subprocess(
        f"""
        import sys
        sys.path.insert(0, {os.path.dirname(env.__file__)!r})
        import standalone_enum_module as m
        assert m.SomeEnum.__class__.__name__ == "pybind11_type"
        """,
        rerun=1,
    )
