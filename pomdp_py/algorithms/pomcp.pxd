from pomdp_py.algorithms.po_uct cimport VNode, RootVNode, POUCT
from pomdp_py.representations.distribution.particles cimport Particles

cdef class VNodeParticles(VNode):
    cdef public Particles belief
cdef class RootVNodeParticles(RootVNode):
    cdef public Particles belief

cdef class POMCP(POUCT):
    pass
