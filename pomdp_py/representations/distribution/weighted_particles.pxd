from pomdp_py.framework.basics cimport GenerativeDistribution

cdef class WeightedParticles(GenerativeDistribution):
    cdef list _particles
    cdef list _values
    cdef list _weights
    cdef dict _hist
    cpdef _get_histogram(WeightedParticles self)
    cpdef mpe(WeightedParticles self)
    cpdef random(WeightedParticles self)
