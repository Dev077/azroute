"""Build the azroute C++ router core in place:  python azroute/setup.py build_ext --inplace"""
import os

import pybind11
from setuptools import Extension, setup

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

setup(
    name="azroute-core",
    ext_modules=[
        Extension(
            "_core",
            ["core/router_core.cpp"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17", "-DNDEBUG"],
        )
    ],
    script_args=["build_ext", "--inplace"],
)
