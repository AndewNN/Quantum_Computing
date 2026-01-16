import sys
from setuptools import setup, Extension
import pybind11

# This helper function is from the official pybind11 docs
# It finds the include directory for pybind11
class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        'ga_solver', 
        ['genetic_solver.cpp'], 
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
        ],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3'],
    ),
]

setup(
    name='ga_solver',
    version='0.1.0',
    description='Genetic Algorithm Bases Initialization C++ Extension',
    long_description='',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5'],
    zip_safe=False,
)