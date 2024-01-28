#!/usr/bin/env python

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import os.path


with open("README.rst", "r") as f:
    long_description = f.read()


# Build cython files as extensions
def build_extensions(pkg_name, major_submodules):
    cwd = os.path.abspath(os.path.dirname(__file__))
    extensions = []
    for subm in major_submodules:
        for f in os.listdir(os.path.join(cwd, pkg_name, subm.replace(".", "/"))):
            if f.endswith(".pyx"):
                filename = os.path.splitext(f)[0]
                ext_name = f"{pkg_name}.{subm}.{filename}"
                ext_path = os.path.join(pkg_name, subm.replace(".", "/"), f)
                extensions.append(Extension(ext_name, [ext_path]))

    return extensions


extensions = build_extensions(
    "pomdp_py",
    [
        "framework",
        "algorithms",
        "utils",
        "representations.distribution",
        "representations.belief",
        "problems.tiger.cythonize",
        "problems.rocksample.cythonize",
    ],
)

setup(
    ext_modules=cythonize(
        extensions, build_dir="build", compiler_directives={"language_level": "3"}
    ),
    packages=find_packages(exclude=["thirdparty", "thirdparty.*"]),
    package_data={
        "pomdp_py": ["*.pxd", "*.pyx", "*.so", "*.c"],
        "pomdp_problems": ["*.pxd", "*.pyx", "*.so", "*.c"],
    },
    zip_safe=False,
)
