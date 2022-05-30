"""This algorithm is PO-UCT (Partially Observable UCT). It is
presented in the POMCP paper :cite:`silver2010monte` as an extension to the UCT
algorithm :cite:`kocsis2006bandit` that combines MCTS and UCB1
for action selection.

In other words, this is just POMCP without particle belief,
but arbitrary belief representation.

This planning strategy, based on MCTS with belief sampling may be referred to as
"belief sparse sampling"; Partially Observable Sparse Sampling (POSS) is
introduced in a recent paper :cite:`lim2019sparse` as an extension of sparse sampling
for MDP by :cite:`kearns2002sparse`; It proposes an extension of POSS
called POWSS (partially observable weighted sparse sampling).  However, this
line of work (POSS, POWSS) is based solely on particle belief
representation. POSS still requires comparing observations exactly for particle
belief update, while POWSS uses weighted particles depending on the observation
distribution.

A note on the exploration constant. Quote from :cite:`gusmao2012towards`:

    "This constant should reflect the agent’s prior knowledge regarding
    the amount of exploration required."

In the POMCP paper, they set this constant following:

    "The exploration constant for POMCP was set to :math:`c = R_{hi} - R_{lo}`,
    where Rhi was the highest return achieved during sample runs of POMCP
    with :math:`c = 0`, and Rlo was the lowest return achieved during sample rollouts"

It is then clear that the POMCP paper is indeed setting this constant
based on prior knowledge. Note the difference between `sample runs` and
`sample rollouts`. But, this is certainly not the only way to obtainx1
the prior knowledge.
"""

from pomdp_py.framework.basics cimport Action, Agent, POMDP, State, Observation,\
    ObservationModel, TransitionModel, GenerativeDistribution, PolicyModel,\
    sample_generative_model
from pomdp_py.framework.planner cimport Planner
from pomdp_py.representations.distribution.particles cimport Particles
from pomdp_py.utils import typ
import copy
import time
import random
import math
from tqdm import tqdm

cdef class TreeNode:
    def __init__(self):
        self.children = {}
    def __getitem__(self, key):
        return self.children.get(key, None)
    def __setitem__(self, key, value):
        self.children[key] = value
    def __contains__(self, key):
        return key in self.children

cdef class QNode(TreeNode):
    def __init__(self, num_visits, value):
        """
        `history_action`: a tuple ((a,o),(a,o),...(a,)). This is only
            used for computing hashses
        """
        self.num_visits = num_visits
        self.value = value
        self.children = {}  # o -> VNode
    def __str__(self):
        return typ.red("QNode") + "(%.3f, %.3f | %s)" % (self.num_visits,
                                                         self.value,
                                                         str(self.children.keys()))
    def __repr__(self):
        return self.__str__()

cdef class VNode(TreeNode):
    def __init__(self, num_visits, **kwargs):
        self.num_visits = num_visits
        self.children = {}  # a -> QNode
    def __str__(self):
        return typ.green("VNode") + "(%.3f, %.3f | %s)" % (self.num_visits,
                                                           self.value,
                                                           str(self.children.keys()))
    def __repr__(self):
        return self.__str__()

    def print_children_value(self):
        for action in self.children:
            print("   action %s: %.3f" % (str(action), self[action].value))

    cpdef argmax(VNode self):
        """argmax(VNode self)
        Returns the action of the child with highest value"""
        cdef Action action, best_action
        cdef float best_value = float("-inf")
        best_action = None
        for action in self.children:
            if self[action].value > best_value:
                best_action = action
                best_value = self[action].value
        return best_action

    @property
    def value(self):
        best_action = max(self.children, key=lambda action: self.children[action].value)
        return self.children[best_action].value


cdef class RootVNode(VNode):
    def __init__(self, num_visits, history):
        VNode.__init__(self, num_visits)
        self.history = history
    @classmethod
    def from_vnode(cls, vnode, history):
        """from_vnode(cls, vnode, history)"""
        rootnode = RootVNode(vnode.num_visits, history)
        rootnode.children = vnode.children
        return rootnode

cdef class ActionPrior:
    """A problem-specific object"""

    cpdef get_preferred_actions(ActionPrior self,
                                State state,
                                tuple history):
        """
        get_preferred_actions(cls, state, history, kwargs)
        Intended as a classmethod.
        This is to mimic the behavior of Simulator.Prior
        and GenerateLegal/GeneratePreferred in David Silver's
        POMCP code.

        Returns a set of tuples, in the form of (action, num_visits_init, value_init)
        that represent the preferred actions.
        In POMCP, this acts as a history-based prior policy,
        and in DESPOT, this acts as a belief-based prior policy.
        For example, given certain state or history, only it
        is possible that only a subset of all actions is legal;
        This is useful when there is domain knowledge that can
        be used as a heuristic for planning. """
        raise NotImplementedError


cdef class RolloutPolicy(PolicyModel):
    cpdef Action rollout(self, State state, tuple history):
        """rollout(self, State state, tuple history=None)"""
        pass

cdef class RandomRollout(RolloutPolicy):
    """A rollout policy that chooses actions uniformly at random from the set of
    possible actions."""
    cpdef Action rollout(self, State state, tuple history):
        """rollout(self, State state, tuple history=None)"""
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

cdef class POUCT(Planner):

    """ POUCT (Partially Observable UCT) :cite:`silver2010monte` is presented in the POMCP
    paper as an extension of the UCT algorithm to partially-observable domains
    that combines MCTS and UCB1 for action selection.

    POUCT only works for problems with action space that can be enumerated.

    __init__(self,
             max_depth=5, planning_time=1., num_sims=-1,
             discount_factor=0.9, exploration_const=math.sqrt(2),
             num_visits_init=1, value_init=0,
             rollout_policy=RandomRollout(),
             action_prior=None, show_progress=False, pbar_update_interval=5)

    Args:
        max_depth (int): Depth of the MCTS tree. Default: 5.
        planning_time (float), amount of time given to each planning step (seconds). Default: -1.
            if negative, then planning terminates when number of simulations `num_sims` reached.
            If both `num_sims` and `planning_time` are negative, then the planner will run for 1 second.
        num_sims (int): Number of simulations for each planning step. If negative,
            then will terminate when planning_time is reached.
            If both `num_sims` and `planning_time` are negative, then the planner will run for 1 second.
        rollout_policy (RolloutPolicy): rollout policy. Default: RandomRollout.
        action_prior (ActionPrior): a prior over preferred actions given state and history.
        show_progress (bool): True if print a progress bar for simulations.
        pbar_update_interval (int): The number of simulations to run after each update of the progress bar,
            Only useful if show_progress is True; You can set this parameter even if your stopping criteria
            is time.
    """

    def __init__(self,
                 max_depth=5, planning_time=-1., num_sims=-1,
                 discount_factor=0.9, exploration_const=math.sqrt(2),
                 num_visits_init=0, value_init=0,
                 rollout_policy=RandomRollout(),
                 action_prior=None, show_progress=False, pbar_update_interval=5):
        self._max_depth = max_depth
        self._planning_time = planning_time
        self._num_sims = num_sims
        if self._num_sims < 0 and self._planning_time < 0:
            self._planning_time = 1.
        self._num_visits_init = num_visits_init
        self._value_init = value_init
        self._rollout_policy = rollout_policy
        self._discount_factor = discount_factor
        self._exploration_const = exploration_const
        self._action_prior = action_prior

        self._show_progress = show_progress
        self._pbar_update_interval = pbar_update_interval

        # to simplify function calls; plan only for one agent at a time
        self._agent = None
        self._last_num_sims = -1
        self._last_planning_time = -1

    @property
    def updates_agent_belief(self):
        return False

    @property
    def last_num_sims(self):
        """Returns the number of simulations ran for the last `plan` call."""
        return self._last_num_sims

    @property
    def last_planning_time(self):
        """Returns the amount of time (seconds) ran for the last `plan` call."""
        return self._last_planning_time

    cpdef public plan(self, Agent agent):
        cdef Action action
        cdef float time_taken
        cdef int sims_count

        self._agent = agent   # switch focus on planning for the given agent
        if not hasattr(self._agent, "tree"):
            self._agent.add_attr("tree", None)
        action, time_taken, sims_count = self._search()
        self._last_num_sims = sims_count
        self._last_planning_time = time_taken
        return action

    cpdef public update(self, Agent agent, Action real_action, Observation real_observation):
        """
        update(self, Agent agent, Action real_action, Observation real_observation)
        Assume that the agent's history has been updated after taking real_action
        and receiving real_observation.
        """
        if not hasattr(agent, "tree") or agent.tree is None:
            print("Warning: agent does not have tree. Have you planned yet?")
            return

        if real_action not in agent.tree\
           or real_observation not in agent.tree[real_action]:
            agent.tree = None  # replan, if real action or observation differs from all branches
        elif agent.tree[real_action][real_observation] is not None:
            # Update the tree (prune)
            agent.tree = RootVNode.from_vnode(
                agent.tree[real_action][real_observation],
                agent.history)
        else:
            raise ValueError("Unexpected state; child should not be None")

    def clear_agent(self):
        self._agent = None  # forget about current agent so that can plan for another agent.
        self._last_num_sims = -1

    cpdef set_rollout_policy(self, RolloutPolicy rollout_policy):
        """
        set_rollout_policy(self, RolloutPolicy rollout_policy)
        Updates the rollout policy to the given one
        """
        self._rollout_policy = rollout_policy

    cpdef _expand_vnode(self, VNode vnode, tuple history, State state=None):
        cdef Action action
        cdef tuple preference
        cdef int num_visits_init
        cdef float value_init

        for action in self._agent.valid_actions(state=state, history=history):
            if vnode[action] is None:
                history_action_node = QNode(self._num_visits_init,
                                            self._value_init)
                vnode[action] = history_action_node

        if self._action_prior is not None:
            # Using action prior; special values are set;
            for preference in \
                self._action_prior.get_preferred_actions(state, history):
                action, num_visits_init, value_init = preference
                history_action_node = QNode(num_visits_init,
                                            value_init)
                vnode[action] = history_action_node


    cpdef _search(self):
        cdef State state
        cdef Action best_action
        cdef int sims_count = 0
        cdef float time_taken = 0
        cdef float best_value
        cdef bint stop_by_sims = self._num_sims > 0
        cdef object pbar

        if self._show_progress:
            if stop_by_sims:
                total = int(self._num_sims)
            else:
                total = self._planning_time
            pbar = tqdm(total=total)

        start_time = time.time()
        while True:
            ## Note: the tree node with () history will have
            ## the init belief given to the agent.
            state = self._agent.sample_belief()
            self._simulate(state, self._agent.history, self._agent.tree,
                           None, None, 0)
            sims_count +=1
            time_taken = time.time() - start_time

            if self._show_progress and sims_count % self._pbar_update_interval == 0:
                if stop_by_sims:
                    pbar.n = sims_count
                else:
                    pbar.n = time_taken
                pbar.refresh()

            if stop_by_sims:
                if sims_count >= self._num_sims:
                    break
            else:
                if time_taken > self._planning_time:
                    if self._show_progress:
                        pbar.n = self._planning_time
                        pbar.refresh()
                    break

        if self._show_progress:
            pbar.close()

        best_action = self._agent.tree.argmax()
        return best_action, time_taken, sims_count

    cpdef _simulate(POUCT self,
                    State state, tuple history, VNode root, QNode parent,
                    Observation observation, int depth):
        if depth > self._max_depth:
            return 0
        if root is None:
            if self._agent.tree is None:
                root = self._VNode(agent=self._agent, root=True)
                self._agent.tree = root
                if self._agent.tree.history != self._agent.history:
                    raise ValueError("Unable to plan for the given history.")
            else:
                root = self._VNode()
            if parent is not None:
                parent[observation] = root
            self._expand_vnode(root, history, state=state)
            rollout_reward = self._rollout(state, history, root, depth)
            return rollout_reward
        cdef int nsteps
        action = self._ucb(root)
        next_state, observation, reward, nsteps = sample_generative_model(self._agent, state, action)
        if nsteps == 0:
            # This indicates the provided action didn't lead to transition
            # Perhaps the action is not allowed to be performed for the given state
            # (for example, the state is not in the initiation set of the option,
            # or the state is a terminal state)
            return reward

        total_reward = reward + (self._discount_factor**nsteps)*self._simulate(next_state,
                                                                               history + ((action, observation),),
                                                                               root[action][observation],
                                                                               root[action],
                                                                               observation,
                                                                               depth+nsteps)
        root.num_visits += 1
        root[action].num_visits += 1
        root[action].value = root[action].value + (total_reward - root[action].value) / (root[action].num_visits)
        return total_reward

    cpdef _rollout(self, State state, tuple history, VNode root, int depth):
        cdef Action action
        cdef float discount = 1.0
        cdef float total_discounted_reward = 0
        cdef State next_state
        cdef Observation observation
        cdef float reward

        while depth < self._max_depth:
            action = self._rollout_policy.rollout(state, history)
            next_state, observation, reward, nsteps = sample_generative_model(self._agent, state, action)
            history = history + ((action, observation),)
            depth += nsteps
            total_discounted_reward += reward * discount
            discount *= (self._discount_factor**nsteps)
            state = next_state
        return total_discounted_reward

    cpdef Action _ucb(self, VNode root):
        """UCB1"""
        cdef Action best_action
        cdef float best_value
        best_action, best_value = None, float('-inf')
        for action in root.children:
            if root[action].num_visits == 0:
                val = float('inf')
            else:
                val = root[action].value + \
                    self._exploration_const * math.sqrt(math.log(root.num_visits + 1) / root[action].num_visits)
            if val > best_value:
                best_action = action
                best_value = val
        return best_action

    cpdef tuple _sample_generative_model(self, State state, Action action):
        '''
        (s', o, r) ~ G(s, a)
        '''
        cdef State next_state
        cdef Observation observation
        cdef float reward

        if self._agent.transition_model is None:
            next_state, observation, reward = self._agent.generative_model.sample(state, action)
        else:
            next_state = self._agent.transition_model.sample(state, action)
            observation = self._agent.observation_model.sample(next_state, action)
            reward = self._agent.reward_model.sample(state, action, next_state)
        return next_state, observation, reward

    def _VNode(self, agent=None, root=False, **kwargs):
        """Returns a VNode with default values; The function naming makes it clear
        that this function is about creating a VNode object."""
        if root:
            return RootVNode(self._num_visits_init, self._agent.history)

        else:
            return VNode(self._num_visits_init)
