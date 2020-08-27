# -*- coding: utf-8 -*-
import os
import sys
import subprocess
from textwrap import dedent

DIR = os.path.abspath(os.path.dirname(__file__))
MAIN_DIR = os.path.dirname(os.path.dirname(DIR))


def test_simple_setup_py(monkeypatch, tmpdir):
    monkeypatch.chdir(tmpdir)
    monkeypatch.syspath_prepend(MAIN_DIR)

    (tmpdir / "setup.py").write_text(
        dedent(
            u"""\
            import sys
            sys.path.append({MAIN_DIR!r})

            from setuptools import setup, Extension
            from pybind11.setup_helpers import BuildExt

            ext_modules = [
                Extension(
                    "simple_setup",
                    sorted(["main.cpp"]),
                    language="c++",
                ),
            ]

            setup(
                name="simple_setup_package",
                cmdclass=dict(build_ext=BuildExt),
                ext_modules=ext_modules
            )
            """
        ).format(MAIN_DIR=MAIN_DIR),
        encoding="ascii",
    )

    (tmpdir / "main.cpp").write_text(
        dedent(
            u"""\
            #include <pybind11/pybind11.h>

            int f(int x) {
                return x * 3;
            }
            PYBIND11_MODULE(simple_setup, m) {
                m.def("f", &f);
            }
            """
        ),
        encoding="ascii",
    )

    subprocess.check_call(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    print(tmpdir.listdir())
    so = list(tmpdir.visit("simple_setup*so"))
    pyd = list(tmpdir.visit("simple_setup*pyd"))
    assert len(so + pyd) == 1
    assert len(list(tmpdir.listdir())) == 4  # two files + output + build_dir

    (tmpdir / "test.py").write_text(
        dedent(
            u"""\
            import simple_setup
            assert simple_setup.f(3) == 9
            """
        ),
        encoding="ascii",
    )

    subprocess.check_call(
        [sys.executable, "test.py"], stdout=sys.stdout, stderr=sys.stderr
    )
