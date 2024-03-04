from pomdp_py.framework.basics cimport GenerativeDistribution

cdef class WeightedParticles(GenerativeDistribution):
    cdef list _particles
    cdef list _values
    cdef list _weights
    cdef str _approx_method
    cdef object _distance_func
    cdef dict _hist
    cdef bint _hist_valid
    cdef bint _frozen
    cdef int _hashcode

    cpdef dict get_histogram(self)

cdef class Particles(WeightedParticles):
    pass
