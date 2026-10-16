"""Build the C++ router core in place as core/_core*.so:  python core/build.py"""
import os

import pybind11
from setuptools import Extension, setup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)     # the extension is named and found relative to the repository root

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
