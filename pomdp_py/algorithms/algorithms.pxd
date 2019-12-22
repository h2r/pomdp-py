from pomdp_py.framework.planner cimport Planner
from pomdp_py.framework.basics cimport Agent, PolicyModel

cdef class TreeNode:
    cdef public int num_visits
    cdef public float value

cdef class QNode(TreeNode):
    pass

cdef class VNode(TreeNode):
    pass

cdef class RootVNode(VNode):
    cdef public tuple history

cdef class POUCT(Planner):
    cdef int _max_depth
    cdef float _planning_time
    cdef int _num_visits_init
    cdef float _value_init
    cdef float _discount_factor
    cdef float _exploration_const
    cdef ActionPrior _action_prior
    cdef RolloutPolicy _rollout_policy
    cdef Agent _agent
    cdef int _last_num_sims    

cdef class RolloutPolicy(PolicyModel):
    cpdef public str rollout(self, VNode vnode, str state)

cdef class RandomRollout(RolloutPolicy):
    pass
    
cdef class ActionPrior:
    cdef str action
    cdef int num_visits_init
    cdef float value_init
    
