# -*- coding: utf-8 -*-
import subprocess
import sys
import os

# These tests must be run explicitly
# They require CMake 3.15+ (--install)

DIR = os.path.abspath(os.path.dirname(__file__))


def test_build_sdist(monkeypatch):

    main_dir = os.path.dirname(os.path.dirname(DIR))
    monkeypatch.chdir(main_dir)

    out = subprocess.check_output([sys.executable, "setup.py", "sdist"])
    if hasattr(out, 'decode'):
        out = out.decode()

    print(out)

    assert 'pybind11/share/cmake/pybind11' in out
    assert 'pybind11/include/pybind11' in out

    assert out.count("copying") == 82
