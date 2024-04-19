# cython: language_level=3

from pomdp_py.algorithms.po_uct cimport QNode
from pomdp_py.algorithms.pomcp cimport POMCP, VNode
from pomdp_py.framework.basics cimport PolicyModel, Action, Agent, State, Observation
from pomdp_py.framework.generalization cimport Response
from pomdp_py.utils.cvec cimport Vector


cdef class CostModel:
    pass


cdef class CCQNode(QNode):
    cdef Vector _cost_value
    cdef Vector _avg_cost_value


cdef class _CCPolicyActionData:
    cdef double _prob
    cdef Vector _cost_value
    cdef Vector _avg_cost_value


cdef class _CCPolicyModel(PolicyModel):
    cdef dict[Action, _CCPolicyActionData] _data
    cdef double _prob_sum

    cdef bint _total_prob_is_not_one(_CCPolicyModel self)
    cpdef void add(_CCPolicyModel self, Action action, double prob, CCQNode node)
    cpdef void clear(_CCPolicyModel self)
    cpdef Vector action_avg_cost(_CCPolicyModel self, Action action)
    cpdef Vector action_cost_value(_CCPolicyModel self, Action action)
    cdef public float probability(_CCPolicyModel self, Action action, State state)
    cdef public Action sample(_CCPolicyModel self, State state)


cdef class CCPOMCP(POMCP):
    cdef double _r_diff
    cdef double _tau
    cdef double _alpha_n
    cdef Vector _lambda
    cdef Vector _cost_constraint
    cdef Response _null_response
    cdef bint _use_random_lambda
    cdef bint _clip_lambda
    cdef double _nu
    cdef list[float] _cost_value_init
    cdef unsigned int _n_constraints
    # Buffers
    cdef Vector _Q_lambda, _Action_UCB
    cdef _CCPolicyModel _greedy_policy_model

    cpdef public Action plan(CCPOMCP self, Agent agent)
    cpdef QNode _create_qnode(self, tuple qnode_params = *)
    cpdef void _greedy_policy(CCPOMCP self, VNode vnode, double explore_const, double nu)
    cdef void _init_lambda_fn(CCPOMCP self)
    cpdef tuple[State, Observation, Response] _sample_generative_model(CCPOMCP self, State state, Action action)
    cpdef _search(CCPOMCP self)
    cpdef Response _simulate(CCPOMCP self, State state, tuple history, VNode root, QNode parent,
                             Observation observation, int depth)
    cdef void _update_cost_constraint(CCPOMCP self, Action sampled_action)


cdef double _compute_visits_ratio(double visits_num, double visits_denom, double explore_const)
cdef double _get_ccqnode_scalar_cost(VNode node, Action action)
