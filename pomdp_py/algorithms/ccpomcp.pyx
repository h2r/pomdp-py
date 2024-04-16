# cython: profile=True

from __future__ import annotations
cimport cython
from libc.math cimport log, sqrt, exp, abs
import math
cimport numpy as cnp
import numpy as np
from pomdp_py.algorithms.po_uct cimport QNode, ActionPrior
from pomdp_py.algorithms.pomcp cimport POMCP
from pomdp_py.framework.basics cimport PolicyModel, Action, Agent, State, Observation
from pomdp_py.framework.generalization cimport (
    Response,
    ResponseAgent,
    sample_generative_model_with_response
)
from pomdp_py.representations.distribution.histogram cimport Histogram
from pomdp_py.representations.distribution.particles cimport Particles
from pomdp_py.utils import typ
from pomdp_py.utils.cvec cimport Vector
from typing import Optional
cnp.import_array()


cdef double DBL_MIN = <double> -1e200
cdef double DBL_MAX = <double> 1e200


cdef class CostModel:
    """
    """

    def probability(
            self,
            cost: float | Vector,
            state: State,
            action: Action,
            next_state: State
    ) -> float:
        """
        probability(self, cost, state, action, next_state)
        Returns the probability of :math:`\Pr(c|s,a,s')`.

        Args:
            cost (float or ~pomdp_py.framework.generalization.Vector): the cost :math:`c`
            state (~pomdp_py.framework.basics.State): the state :math:`s`
            action (~pomdp_py.framework.basics.Action): the action :math:`a`
            next_state (State): the next state :math:`s'`

        Returns:
            float: the probability :math:`\Pr(c|s,a,s')`
        """
        raise NotImplementedError

    def sample(
            self,
            state: State,
            action: Action,
            next_state: State,
            **kwargs,
    ) -> float | Vector:
        """
        sample(self, state, action, next_state)
        Returns a cost randomly sampled according to the
        distribution of this cost model. This is required for cost-aware planners.

        Args:
            state (~pomdp_py.framework.basics.State): the next state :math:`s`
            action (~pomdp_py.framework.basics.Action): the action :math:`a`
            next_state (State): the next state :math:`s'`

        Returns:
            float or ~pomdp_py.framework.generalization.Vector: the cost :math:`c`
        """
        raise NotImplementedError

    def argmax(self, state: State, action: Action, next_state: State) -> float | Vector:
        """
        argmax(self, state, action, next_state)
        Returns the most likely cost. This is optional.
        """
        raise NotImplementedError

    def get_distribution(self, state: State, action: Action, next_state: State):
        """
        get_distribution(self, state, action, next_state)
        Returns the underlying distribution of the model.
        """
        raise NotImplementedError


cdef class CCQNode(QNode):
    def __init__(
        self,
        num_visits: int,
        value: float,
        cost_value: list[float],
    ) -> None:
        super().__init__(num_visits=num_visits, value=value)
        if len(cost_value) == 0:
            raise ValueError("len(cost_value) must be positive.")
        self._cost_value = Vector(cost_value)
        self._avg_cost_value = Vector.null(self._cost_value.len())

    @property
    def avg_cost_value(self) -> Vector:
        return self._avg_cost_value.copy()

    @avg_cost_value.setter
    def avg_cost_value(self, avg_cost_value: Vector) -> None:
        if not isinstance(avg_cost_value, Vector):
            raise TypeError(
                "avg_cost_value must be type Vector, "
                f"but got {type(avg_cost_value)}."
            )
        self._avg_cost_value = avg_cost_value.copy()

    @property
    def cost_value(self) -> Vector:
        return self._cost_value.copy()

    @cost_value.setter
    def cost_value(self, cost_value: Vector) -> None:
        if not isinstance(cost_value, Vector):
            raise TypeError(
                "cost_value must be type Vector, "
                f"but got {type(cost_value)}."
            )
        self._cost_value = cost_value.copy()

    def __str__(self) -> str:
        return (
            typ.red("CCQNode")
            + f"(n={self.num_visits}, v={self.value:.3f}, c={self.cost_value} "
            + f"c_bar={self.avg_cost_value} | children=[{', '.join(list(self.children.keys()))}])"
        )

cdef class _CCPolicyActionData:
    def __init__(self, double prob, Vector cost_value, Vector avg_cost_value):
        self._prob = prob
        self._cost_value = cost_value.copy()
        self._avg_cost_value = avg_cost_value.copy()

    @property
    def prob(self) -> float:
        return self._prob

    @property
    def cost_value(self) -> Vector:
        return self._cost_value

    @property
    def avg_cost_value(self) -> Vector:
        return self._avg_cost_value


cdef class _CCPolicyModel(PolicyModel):
    def __init__(self, dict[Action, _CCPolicyActionData] data_dict) -> None:
        super().__init__()
        cdef Action action
        cdef _CCPolicyActionData datum
        cdef double prob_sum = 0.0

        for action, datum in data_dict.items():
            if not isinstance(action, Action):
                raise TypeError("action must be type Action.")
            prob_sum += datum.prob

        if prob_sum != 1.0:
            raise ValueError(f"The probabilities must sum to 1.0, but got {prob_sum}.")
        self._data = data_dict.copy()

    cpdef Vector action_avg_cost(_CCPolicyModel self, Action action):
        if action not in self._data:
            raise KeyError(f"The action {action} is not exist in this policy model.")
        return self._data[action].cost_value.copy()

    cpdef Vector action_cost_value(_CCPolicyModel self, Action action):
        if action not in self._data:
            raise KeyError(f"The action {action} is not exist in this policy model.")
        return self._data[action].avg_cost_value.copy()

    cdef public float probability(_CCPolicyModel self, Action action, State state):
        if action not in self._data:
            raise KeyError(f"The action {action} is not exist in this policy model.")
        return self._data[action].prob

    cdef public Action sample(_CCPolicyModel self, State state):
        return np.random.choice(np.array(list(self._data.keys()), dtype=object))

    def get_all_actions(self, state: Optional[State] = None, history: Optional[tuple] = None):
        return list(self._data.keys())


cdef class CCPOMCP(POMCP):
    """
    The cost-constrained POMCP (CCPOMCP) is POMCP + cost constraints.
    The current implementation assumes the cost constraint is 1D.
    """

    def __init__(
        self,
        r_diff: float,
        tau: float,
        alpha_n: float,
        cost_constraint: list[float] | float,
        clip_lambda: bool = True,
        nu: float = 1.0,
        max_depth: int = 5,
        planning_time: float = -1.0,
        num_sims: int = -1,
        discount_factor: float = 0.9,
        exploration_const: float = math.sqrt(2.0),
        num_visits_init: int = 0,
        value_init: int = 0,
        cost_value_init: Optional[list[float] | float] = None,
        use_random_lambda: bool = True,
        rollout_policy: Optional[PolicyModel] = None,
        action_prior: Optional[ActionPrior] = None,
        show_progress: bool = False,
        pbar_update_interval: int = 5
    ):
        super(CCPOMCP, self).__init__(
            max_depth=max_depth,
            planning_time=planning_time,
            num_sims=num_sims,
            discount_factor=discount_factor,
            exploration_const=exploration_const,
            num_visits_init=num_visits_init,
            value_init=value_init,
            rollout_policy=rollout_policy,
            action_prior=action_prior,
            show_progress=show_progress,
            pbar_update_interval=pbar_update_interval
        )
        # Sanity checks and set the parameters.
        if not isinstance(r_diff, float):
            raise TypeError(f"r_diff must be type float, but got {type(r_diff)}.")
        if r_diff < 0.0:
            raise ValueError("r_diff must be a non-negative float.")
        if not isinstance(tau, float):
            raise TypeError(f"tau must be type float, but got {type(tau)}.")
        if not isinstance(alpha_n, float):
            raise TypeError(f"alpha_n must be type float, but got {type(alpha_n)}.")
        if alpha_n < 0.0 or 1.0 < alpha_n:
            raise ValueError("alpha_n must be in range [0.0, 1.0].")
        if not isinstance(cost_constraint, (list, float)):
            raise TypeError(
                "cost_constraint must be a Vector or float "
                f"but got type {type(cost_constraint)}."
            )
        if not isinstance(clip_lambda, bool):
            raise TypeError(
                f"clip_lambda must be a Boolean, but got type {type(clip_lambda)}."
            )
        if not isinstance(nu, float):
            raise TypeError(f"nu must be type float, but got {type(nu)}.")
        if not isinstance(use_random_lambda, bool):
            raise TypeError(
                "use_random_lambda must be type bool, "
                f"but got {type(use_random_lambda)}."
            )

        if cost_value_init is not None:
            if not isinstance(cost_value_init, (list, float)):
                raise TypeError(
                    "cost_value_init must be type Vector or float, "
                    f"but got {type(cost_value_init)}."
                )
            if type(cost_value_init) != type(cost_constraint):
                raise TypeError(
                    "cost_value_init and cost_constraint must be the same type."
                )

        # Initialize lambda, cost constraint, and cost value init.
        if isinstance(cost_constraint, list):
            self._n_constraints = len(cost_constraint)
            if len(cost_value_init) != len(cost_value_init):
                raise ValueError(
                    "The cost constraint and cost value init must have the same length."
                )
        else:
            self._n_constraints = 1
            cost_constraint = [cost_constraint]
            cost_value_init = [cost_value_init] if cost_value_init is not None else [0.0]

        self._lambda = Vector.null(self._n_constraints)
        self._cost_value_init = list(cost_value_init)
        self._cost_constraint = Vector(cost_constraint)
        self._r_diff = <double> r_diff
        self._tau = <double> tau
        self._alpha_n = <double> alpha_n
        self._clip_lambda = <bint> clip_lambda
        self._nu = <double> nu
        self._use_random_lambda = <bint> use_random_lambda

    cpdef public Action plan(CCPOMCP self, Agent agent):
        cdef Action action
        cdef _CCPolicyModel policy_dist
        cdef double time_taken
        cdef int sims_count

        if not isinstance(agent.belief, Particles):
            raise TypeError(
                "Agent's belief is not represented in particles. "
                "CCPOMCP not usable. Please convert it to particles."
            )

        if self._rollout_policy is None:
            raise ValueError(
                "rollout_policy unset. Please call set_rollout_policy, "
                "or pass in a rollout_policy upon initialization."
            )

        if not isinstance(agent, ResponseAgent):
            raise TypeError(
                f"agent must be type ResponseAgent, but got type {type(agent)}."
            )

        # Set the current agent being used for planning.
        self._agent = agent
        self._null_response = self._agent.response_model.null_response()
        if not hasattr(self._agent, "tree"):
            self._agent.add_attr("tree", None)

        # Then get the policy distribution, sample from it,
        # and update the cost constraint.
        policy_dist, time_taken, sims_count = self._search()
        action = policy_dist.sample(state=None)
        self._update_cost_constraint(policy_dist, action)

        # Update stats.
        self._last_num_sims = sims_count
        self._last_planning_time = time_taken

        return action

    cpdef _expand_vnode(
        CCPOMCP self,
        VNode vnode,
        tuple history,
        State state = None,
    ):
        cdef Action action

        for action in self._agent.valid_actions(state=state, history=history):
            if vnode[action] is None:
                vnode[action] = CCQNode(
                    self._num_visits_init, self._value_init, self._cost_value_init
                )

        if self._action_prior is not None:
            # Using action prior; special values are set;
            for preference in self._action_prior.get_preferred_actions(state, history):
                action, num_visits_init, value_init = preference
                vnode[action] = CCQNode(
                    self._num_visits_init, self._value_init, self._cost_value_init
                )

    @cython.boundscheck(False)
    cpdef _CCPolicyModel _greedy_policy(
        CCPOMCP self,
        VNode vnode,
        double explore_const,
        double nu,
    ):
        cdef list[Action] action_list = list(vnode.children.keys())
        cdef int n_actions = len(action_list)

        if n_actions == 0:
            raise RuntimeError("The vnode has no visited actions?!")

        # Compute Q_lambda.
        cdef double n_ccqnode_visits
        cdef double best_q_lambda = DBL_MIN
        cdef int best_q_index = 0
        cdef double logN = log(<double> vnode.num_visits + 1)
        cdef double q_value
        cdef Vector Q_lambda, Action_UCB
        cdef CCQNode ccqnode
        cdef Action action
        cdef int i = 0

        Q_lambda = Vector.null(n_actions)
        Action_UCB = Vector.null(n_actions)

        for i in range(n_actions):
            ccqnode = vnode[action_list[i]]
            q_value = ccqnode.value - self._lambda.dot(ccqnode.cost_value)

            if ccqnode.num_visits > 0:
                n_ccqnode_visits = <double> ccqnode.num_visits + 1.0
                q_value += _compute_visits_ratio(
                    logN,
                    n_ccqnode_visits,
                    explore_const
                )
                Action_UCB[i] = _compute_visits_ratio(
                    log(n_ccqnode_visits),
                    n_ccqnode_visits
                )

            if q_value > best_q_lambda:
                best_q_lambda = q_value
                best_q_index = i

            Q_lambda[i] = q_value

        # Compute a*, the best action(s).
        cdef list[Action] best_action_list = list()
        cdef double best_ucb_add = Action_UCB[best_q_index]
        cdef double ucb_add, q_value_diff
        cdef bint add_to_best_action_list = False

        q_value = 0.0
        best_q_lambda = Q_lambda[best_q_index]

        for i in range(n_actions):
            action = action_list[i]
            ccqnode = vnode[action]
            q_value = Q_lambda[i]

            if q_value == best_q_lambda:
                add_to_best_action_list = True

            else:
                q_value_diff = abs(q_value - best_q_lambda)
                ucb_add = nu * (Action_UCB[i] + best_ucb_add)
                if q_value_diff <= ucb_add and action not in best_action_list:
                    add_to_best_action_list = True

            if add_to_best_action_list:
                best_action_list.append(action)

        # Find the policy.
        cdef int n_best_actions = len(best_action_list)
        cdef Action action_min, action_max
        cdef CCQNode ccqnode_min, ccqnode_max
        cdef double cost_constraint_scalar = self._cost_constraint[0]
        cdef double max_cost_value, min_cost_value, min_prob, cost_value
        cdef dict[Action, _CCPolicyActionData] data

        if n_best_actions == 0:
            raise RuntimeError("No best actions were found?!")

        elif n_best_actions == 1:
            action = best_action_list[0]
            data = {
                action: _CCPolicyActionData(
                    1.0,
                    vnode[action].cost_value,
                    vnode[action].avg_cost_value
                )
            }

        else:
            # TODO: Implement linear programming to handle multiple constraints.
            #       The code below can only handle ONE constraint.
            if self._cost_constraint.len() > 1:
                raise NotImplementedError(
                    f"This algorithm can only handle one constraint for now."
                )
            # if self._lambda[0] <= 0.0:
            #     raise RuntimeError(
            #         "The scalar lambda must be positive to continue. "
            #         "See the Appendix G in the Supplementary Materials for the paper "
            #         "titled 'Monte-Carlo Tree Search for Constrained POMDPs' "
            #         "by Lee et. al (2018)."
            #     )

            # Find a_max and a_min, the actions with the max and min scalar costs
            # from the list of best actions.
            max_cost_value = DBL_MIN
            min_cost_value = DBL_MAX

            for i in range(n_best_actions):
                action = best_action_list[i]
                cost_value = _get_ccqnode_scalar_cost(vnode, action)

                if cost_value < min_cost_value:
                    action_min = action
                    min_cost_value = cost_value

                if cost_value > max_cost_value:
                    action_max = action
                    max_cost_value = cost_value

            # Sanity checks.
            if max_cost_value == DBL_MIN:
                raise RuntimeError(
                    f"Max cost value ({max_cost_value}) must be more than {DBL_MIN}. "
                    f"Note: there are {n_best_actions} best actions. An error exists!"
                )
            if min_cost_value == DBL_MAX:
                raise RuntimeError(
                    f"Min cost value ({min_cost_value}) must be less than {DBL_MAX}. "
                    f"Note: there are {n_best_actions} best actions. An error exists!"
                )

            if max_cost_value <= cost_constraint_scalar or action_min == action_max:
                data = {
                    action_max: _CCPolicyActionData(
                        1.0,
                        vnode[action_max].cost_value,
                        vnode[action_max].avg_cost_value
                    )
                }

            elif min_cost_value <= cost_constraint_scalar:
                data = {
                    action_min: _CCPolicyActionData(
                        1.0,
                        vnode[action_min].cost_value,
                        vnode[action_min].avg_cost_value
                    )
                }

            else:
                min_prob = (
                    (max_cost_value - cost_constraint_scalar)
                    / (max_cost_value - min_cost_value)
                )

                data = {
                    action_min: _CCPolicyActionData(
                        min_prob,
                        vnode[action_min].cost_value,
                        vnode[action_min].avg_cost_value
                    ),
                    action_max: _CCPolicyActionData(
                        1.0 - min_prob,
                        vnode[action_max].cost_value,
                        vnode[action_max].avg_cost_value
                    ),
                }

        return _CCPolicyModel(data)

    cdef void _init_lambda_fn(CCPOMCP self):
        if self._use_random_lambda:
            self._lambda = Vector(
                np.random.uniform(
                    0.00001,
                    1.0,
                    size=self._cost_constraint.len()
                ).tolist()
            )

        else:
            self._lambda = Vector.null(self._n_constraints)

    cpdef _perform_simulation(self, state):
        super(CCPOMCP, self)._perform_simulation(state=state)

        # Sample using the greedy policy. This greedy policy corresponds to the first
        # call in the search(h_0) function.
        policy_dist = self._greedy_policy(
            self._agent.tree,
            0.0,
            0.0,
        )
        action = policy_dist.sample(state=state)

        # Update lambda.
        self._lambda = self._lambda + self._alpha_n * (
                self._agent.tree[action].cost_value - self._cost_constraint
        )
        if self._clip_lambda:
            lambda_vec_max = self._r_diff / (
                    self._tau * (1.0 - self._discount_factor)
            )
            self._lambda = Vector.clip(self._lambda, 0.0, lambda_vec_max)

    cpdef _rollout(self, State state, tuple history, VNode root, int depth):
        cdef Action action
        cdef float discount = 1.0
        cdef State next_state
        cdef Observation observation
        cdef Response response, total_discounted_response
        cdef int nsteps

        total_discounted_response = self._null_response
        while depth < self._max_depth:
            action = self._rollout_policy.rollout(state, history)
            next_state, observation, response, nsteps = (
                sample_generative_model_with_response(
                    self._agent.transition_model,
                    self._agent.observation_model,
                    self._agent.response_model,
                    state,
                    action,
                    self._null_response,
                )
            )
            history = history + ((action, observation),)
            depth += nsteps
            total_discounted_response = (
                    total_discounted_response + (response * discount)
            )
            discount *= (self._discount_factor ** nsteps)
            state = next_state
        return total_discounted_response

    cpdef _search(CCPOMCP self):
        cdef Action action
        cdef double time_taken
        cdef int sims_count
        cdef PolicyModel policy_dist

        # Initialize the lambda vector.
        self._init_lambda_fn()

        # Run the _search(...) method in the super class.
        action, time_taken, sims_count = super(CCPOMCP, self)._search()

        # After the search times out, create a policy using the greedy method.
        # This greedy policy corresponds to the last call in the search(h_0) function.
        policy_dist = self._greedy_policy(
            self._agent.tree,
            0.0,
            self._nu,
        )
        return policy_dist, time_taken, sims_count

    cpdef Response _simulate(
        CCPOMCP self,
        State state,
        tuple history,
        VNode root,
        QNode parent,
        Observation observation,
        int depth
    ):
        cdef Response response, total_response
        cdef int nsteps = 1
        cdef Action action
        cdef State next_state
        cdef _CCPolicyModel policy_dist

        if depth > self._max_depth:
            return self._null_response

        if root is None:
            if self._agent.tree is None:
                root = self._VNode(root=True)
                self._agent.tree = root
                if self._agent.tree.history != self._agent.history:
                    raise ValueError("Unable to plan for the given history.")

            else:
                root = self._VNode()

            if parent is not None:
                parent[observation] = root

            self._expand_vnode(root, history, state=state)
            response = self._rollout(state, history, root, depth)
            return response

        # This greedy policy corresponds to the call in the simulate(s, h, d) function
        # in the paper.
        policy_dist = self._greedy_policy(
            root,
            self._exploration_const,
            self._nu
        )
        action = policy_dist.sample(state=state)
        next_state, observation, response, nsteps = (
            sample_generative_model_with_response(
                self._agent.transition_model,
                self._agent.observation_model,
                self._agent.response_model,
                state,
                action,
                self._null_response,
            )
        )

        if nsteps == 0:
            return response

        total_response = (
            response
            + (self._discount_factor ** nsteps)
            * self._simulate(
                next_state,
                history + ((action, observation),),
                root[action][observation],
                root[action],
                observation,
                depth + nsteps
            )
        )

        root.num_visits += 1
        root[action].num_visits += 1
        root[action].value = (
                root[action].value
                + (total_response.reward - root[action].value) / root[action].num_visits
        )

        root[action].cost_value = (
                root[action].cost_value
                + (total_response.cost - root[action].cost_value) / root[action].num_visits
        )

        root[action].avg_cost_value = (
                root[action].avg_cost_value
                + (response.cost - root[action].avg_cost_value) / root[action].num_visits
        )

        if depth == 1 and root is not None:
            root.belief.add(state)

        return total_response

    cdef void _update_cost_constraint(
        CCPOMCP self,
        _CCPolicyModel policy_dist,
        Action sampled_action
    ):
        cdef double action_prob, prob_prime
        cdef Vector chat_minus_avg_cost, action_avg_cost, cost_value, cost_sum
        cdef Action action_prime
        cdef list[Action] action_prime_list
        cdef int i = 0

        action_prob = policy_dist.probability(
            action=sampled_action,
            state=None
        )
        action_avg_cost = policy_dist.action_avg_cost(sampled_action)
        self._cost_constraint -= (action_prob * action_avg_cost)

        if action_prob < 1.0:
            cost_sum = Vector.null(self._n_constraints)
            action_prime_list = policy_dist.get_all_actions()
            for i in range(len(action_prime_list)):
                action_prime = action_prime_list[i]
                if action_prime == sampled_action:
                    continue

                prob_prime = policy_dist.probability(
                    action=action_prime,
                    state=self._agent.history
                )
                cost_value = policy_dist.action_cost_value(sampled_action)
                cost_sum[i] += (prob_prime * cost_value)
            self._cost_constraint -= cost_sum
        self._cost_constraint /= (self._discount_factor * action_prob)


cdef double _compute_visits_ratio(
        double visits_num,
        double visits_denom,
        double explore_const = 1.0,
):
    if visits_denom == 0.0:
        return DBL_MIN
    else:
        return explore_const * sqrt(visits_num / visits_denom)


cdef double _get_ccqnode_scalar_cost(
    VNode node,
    Action action
):
    if action not in node:
        raise KeyError(f"Action {action} does not exist in node.")
    return node[action].cost_value[0]
