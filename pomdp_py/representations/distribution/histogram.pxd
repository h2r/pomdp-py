from pomdp_py.framework.basics cimport GenerativeDistribution

cdef class Histogram(GenerativeDistribution):
    cdef dict _histogram
