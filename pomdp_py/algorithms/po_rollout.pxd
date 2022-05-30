from pomdp_py.framework.basics cimport Action, State, Observation, Agent
from pomdp_py.framework.planner cimport Planner
from pomdp_py.algorithms.po_uct cimport RolloutPolicy, ActionPrior

cdef class PORollout(Planner):

    cdef int _num_sims
    cdef int _max_depth
    cdef RolloutPolicy _rollout_policy
    cdef ActionPrior _action_prior
    cdef float _discount_factor
    cdef bint _particles
    cdef Agent _agent
    cdef float _last_best_reward

    cpdef _search(self)
    cpdef _rollout(self, State state, int depth)
    cpdef update(self, Agent agent, Action real_action, Observation real_observation,
                 state_transform_func=*)

    cpdef set_rollout_policy(self, RolloutPolicy rollout_policy)
