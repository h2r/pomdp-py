#!/usr/bin/env python

from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize

with open("README.rst", 'r') as f:
    long_description = f.read()

setup(name='pomdp-py',
      packages=find_packages(),
      version='1.2.4',
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
      author='Kaiyu Zheng',
      author_email='kaiyutony@gmail.com',
      keywords = ['Partially Observable Markov Decision Process', 'POMDP'],
      ext_modules=cythonize(['pomdp_py/algorithms/po_rollout.pyx',
                             'pomdp_py/algorithms/po_uct.pyx',
                             'pomdp_py/algorithms/pomcp.pyx',
                             'pomdp_py/algorithms/value_iteration.pyx',
                             'pomdp_py/framework/oopomdp.pyx',
                             'pomdp_py/framework/planner.pyx',
                             'pomdp_py/framework/basics.pyx',
                             'pomdp_py/representations/distribution/particles.pyx',
                             'pomdp_py/representations/distribution/histogram.pyx',
                             'pomdp_py/representations/distribution/gaussian.pyx',
                             'pomdp_py/representations/belief/particles.pyx',
                             'pomdp_problems/tiger/cythonize/tiger_problem.pyx',
                             'pomdp_problems/rocksample/cythonize/rocksample_problem.pyx'],
                            build_dir="build", compiler_directives={'language_level' : "3"})
     )
