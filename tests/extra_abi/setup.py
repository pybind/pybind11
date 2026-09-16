from __future__ import annotations

import os

from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension

name = os.environ["EXAMPLE_NAME"]
assert name in {"pet", "dog"}

ext = Pybind11Extension(name, [f"{name}.cpp"], include_dirs=["."], cxx_std=17)

setup(name=name, version="0.0.0", ext_modules=[ext])
