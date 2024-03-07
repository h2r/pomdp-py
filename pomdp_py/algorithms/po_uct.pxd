from pomdp_py.framework.planner cimport Planner
from pomdp_py.framework.basics cimport Agent, PolicyModel, Action, State, Observation

cdef class TreeNode:
    cdef public dict children
    cdef public int num_visits
    cdef public float value

cdef class QNode(TreeNode):
    pass

cdef class VNode(TreeNode):
    cpdef argmax(VNode self)

cdef class RootVNode(VNode):
    cdef public tuple history

cdef class POUCT(Planner):
    cdef int _max_depth
    cdef float _planning_time
    cdef int _num_sims
    cdef int _num_visits_init
    cdef float _value_init
    cdef float _discount_factor
    cdef float _exploration_const
    cdef ActionPrior _action_prior
    cdef RolloutPolicy _rollout_policy
    cdef Agent _agent
    cdef int _last_num_sims
    cdef float _last_planning_time
    cdef bint _show_progress
    cdef int _pbar_update_interval

    cpdef _search(self)
    cdef _initialize_progress_bar(self)
    cpdef _perform_simulation(self, state)
    cdef bint _should_stop(self, int sims_count, double start_time)
    cdef _update_progress(self, pbar, int sims_count, double start_time)
    cdef _finalize_progress_bar(self, pbar)

    cpdef _simulate(POUCT self,
                    State state, tuple history, VNode root, QNode parent,
                    Observation observation, int depth)

    cpdef _expand_vnode(self, VNode vnode, tuple history, State state=*)
    cpdef _rollout(self, State state, tuple history, VNode root, int depth)
    cpdef Action _ucb(self, VNode root)
    cpdef tuple _sample_generative_model(self, State state, Action action)
    cpdef set_rollout_policy(self, RolloutPolicy rollout_policy)

cdef class RolloutPolicy(PolicyModel):
    cpdef Action rollout(self, State state, tuple history)

cdef class RandomRollout(RolloutPolicy):
    pass

cdef class ActionPrior:
    cpdef get_preferred_actions(ActionPrior self, State state, tuple history)
