#!/usr/bin/env python

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import os.path


with open("README.rst", 'r') as f:
    long_description = f.read()

extensions = [
    Extension("pomdp_py.algorithms", ["pomdp_py/algorithms/*.pyx"]),
    Extension("pomdp_py.framework", ["pomdp_py/framework/*.pyx"]),
    Extension("pomdp_py.representations.distribution", ["pomdp_py/representations/distribution/*.pyx"]),
    Extension("pomdp_py.representations.belief", ["pomdp_py/representations/belief/*.pyx"]),
    Extension("pomdp_problems.tiger.cythonize", ["pomdp_problems/tiger/cythonize/tiger_problem.pyx"]),
    Extension("pomdp_problems.rocksample.cythonize", ["pomdp_problems/rocksample/cythonize/rocksample_problem.pyx"])
]

setup(name='pomdp-py',
      packages=find_packages(),
      version='1.2.4.5',
      description='Python POMDP Library.',
      long_description=long_description,
      long_description_content_type="text/x-rst",
      install_requires=[
          'Cython',
          'numpy',
          'scipy',
          'matplotlib',
          'pygame',        # for some tests
          'opencv-python',  # for some tests
          'networkx',
          'pygraphviz'
      ],
      license="MIT",
      author='Kaiyu Zheng',
      author_email='kzheng10@cs.brown.edu',
      keywords = ['Partially Observable Markov Decision Process', 'POMDP'],
      ext_modules=cythonize(extensions,
                            build_dir="build",
                            compiler_directives={'language_level' : "3"}),
      package_data={"pomdp_py": ["*.pxd", "*.pyx"],
                    "pomdp_problems": ["*.pxd", "*.pyx"]},
      zip_safe=False
)
