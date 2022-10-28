import os
import sys

from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension

name = os.environ["EXAMPLE_NAME"]
assert name in {"pet", "dog"}


ext = Pybind11Extension(
    name,
    [f"{name}.cpp"],
    include_dirs=["."],
    cxx_std=11,
    extra_compile_args=["/d2FH4-"] if sys.platform.startswith("win32") else [],
)

setup(name=name, version="0.0.0", ext_modules=[ext])
