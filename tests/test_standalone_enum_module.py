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
        assert int(m.SomeEnum.value1) == 0
        assert int(m.SomeEnum.value2) == 1
        """,
        rerun=1,
    )
