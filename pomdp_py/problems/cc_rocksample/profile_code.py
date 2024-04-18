import pstats, cProfile

import pyximport
pyximport.install()

import cc_rocksample_problem

cProfile.runctx("cc_rocksample_problem.main()", globals(), locals(), "fastProfile.prof")

s = pstats.Stats("fastProfile.prof")
s.strip_dirs().sort_stats("tottime").print_stats()
