import pstats, cProfile

import pyximport
pyximport.install()

import rocksample_problem

cProfile.runctx("rocksample_problem.main()", globals(), locals(), "fastProfile.prof")

s = pstats.Stats("fastProfile.prof")
s.strip_dirs().sort_stats("tottime").print_stats()
