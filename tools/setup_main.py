#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Setup script for PyPI - placed in the sdist

from setuptools import setup

setup(
    name="pybind11",
    packages=[
        "pybind11",
        "pybind11.include.pybind11",
        "pybind11.include.pybind11.detail",
        "pybind11.share.cmake.pybind11",
    ],
    package_data={
        "pybind11.include.pybind11": ["*.h"],
        "pybind11.include.pybind11.detail": ["*.h"],
        "pybind11.share.cmake.pybind11": ["*.cmake"],
    },
    extras_require={
        "global": ["pybind11_global==2.6.0"]
        },
    entry_points={
        "console_scripts": [
             "pybind11-config = pybind11.__main__:main",
        ]
    }
)
