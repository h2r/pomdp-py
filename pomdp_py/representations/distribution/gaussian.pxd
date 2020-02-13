from pomdp_py.framework.basics cimport GenerativeDistribution

cdef class Gaussian(GenerativeDistribution):
    cdef list _mean
    cdef list _cov
