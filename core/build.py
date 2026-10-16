"""Build the C++ router core in place as core/_core*.so:  python setup.py"""
import os

import pybind11
from setuptools import Extension, setup

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

setup(
    name="azroute-core",
    ext_modules=[
        Extension(
            "core._core",
            ["core/router_core.cpp"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17", "-DNDEBUG"],
        )
    ],
    script_args=["build_ext", "--inplace"],
)
