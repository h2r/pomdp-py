from pomdp_py.algorithms.po_uct cimport VNode, RootVNode
from pomdp_py.representations.distribution.particles cimport Particles

cdef class VNodeParticles(VNode):
    cdef public Particles belief
cdef class RootVNodeParticles(RootVNode):
    cdef public Particles belief

