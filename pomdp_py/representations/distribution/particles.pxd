from pomdp_py.framework.basics cimport GenerativeDistribution

cdef class Particles(GenerativeDistribution):
    cdef list _particles
