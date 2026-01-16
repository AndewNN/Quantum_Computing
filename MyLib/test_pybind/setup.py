from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "pybind_example",
        ["pybind_example.cpp"], 
        include_dirs=[pybind11.get_include()],
        language="c++", 
    )
]

setup(
    name="pybind_example",
    version="0.1",
    ext_modules=ext_modules,
)