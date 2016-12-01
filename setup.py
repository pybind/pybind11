#!/usr/bin/env python

# Setup script for PyPI; use CMakeFile.txt to build extension modules

from setuptools import setup
from pybind11 import __version__

setup(
    name='pybind11',
    version=__version__,
    description='Seamless operability between C++11 and Python',
    author='Wenzel Jakob',
    author_email='wenzel.jakob@epfl.ch',
    url='https://github.com/wjakob/pybind11',
    download_url='https://github.com/wjakob/pybind11/tarball/v' + __version__,
    packages=['pybind11'],
    license='BSD',
    headers=[
        'include/pybind11/attr.h',
        'include/pybind11/cast.h',
        'include/pybind11/chrono.h',
        'include/pybind11/common.h',
        'include/pybind11/complex.h',
        'include/pybind11/descr.h',
        'include/pybind11/eigen.h',
        'include/pybind11/eval.h',
        'include/pybind11/functional.h',
        'include/pybind11/numpy.h',
        'include/pybind11/operators.h',
        'include/pybind11/options.h',
        'include/pybind11/pybind11.h',
        'include/pybind11/pytypes.h',
        'include/pybind11/stl.h',
        'include/pybind11/stl_bind.h',
        'include/pybind11/typeid.h',
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
        'Programming Language :: C++',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'License :: OSI Approved :: BSD License',
    ],
    keywords='C++11, Python bindings',
    long_description="""pybind11 is a lightweight header library that exposes
C++ types in Python and vice versa, mainly to create Python bindings of
existing C++ code. Its goals and syntax are similar to the excellent
Boost.Python library by David Abrahams: to minimize boilerplate code in
traditional extension modules by inferring type information using compile-time
introspection.

The main issue with Boost.Python-and the reason for creating such a similar
project-is Boost. Boost is an enormously large and complex suite of utility
libraries that works with almost every C++ compiler in existence. This
compatibility has its cost: arcane template tricks and workarounds are
necessary to support the oldest and buggiest of compiler specimens. Now that
C++11-compatible compilers are widely available, this heavy machinery has
become an excessively large and unnecessary dependency.

Think of this library as a tiny self-contained version of Boost.Python with
everything stripped away that isn't relevant for binding generation. Without
comments, the core header files only require ~2.5K lines of code and depend on
Python (2.7 or 3.x) and the C++ standard library. This compact implementation
was possible thanks to some of the new C++11 language features (specifically:
tuples, lambda functions and variadic templates). Since its creation, this
library has grown beyond Boost.Python in many ways, leading to dramatically
simpler binding code in many common situations.""")
