# This algorithm is PO-UCT (Partially Observable UCT). It is
# presented in the POMCP paper as an extension to the UCT
# algorithm [1] that combines MCTS and UCB1 for action selection.
#     
# In other words, this is just POMCP without particle belief,
# but arbitrary belief representation.
#
# This planning strategy, based on MCTS with belief sampling
# may be referred to as "belief sparse sampling"; Partially
# Observable Sparse Sampling (POSS) is introduced in a
# recent paper [2] as an extension of sparse sampling for MDP by
# (Kearns et al. 2002); [2] proposes an extension of POSS
# called POWSS (partially observable weighted sparse sampling).
# However, this line of work (POSS, POWSS) is based solely on particle
# belief representation. POSS still requires comparing observations
# exactly for particle belief update, while POWSS uses weighted
# particles depending on the observation distribution.
#
# [1] Bandit based Monte-Carlo Planning, by L. Kocsis and C. Szepesv´ari
# [2] Sparse tree search optimality guarantees n POMDPs with
#     continuous observation spaces, by M. Lim, C. Tomlin, Z. Sunberg.

from pomdp_py.framework.basics cimport Action, Agent, POMDP, State, Observation,\
    ObservationModel, TransitionModel, GenerativeDistribution, PolicyModel,\
    sample_generative_model
from pomdp_py.framework.planner cimport Planner
from pomdp_py.representations.distribution.particles cimport Particles
import copy
import time
import random
import math

cdef class TreeNode:
    def __init__(self):
        self.children = {}
    def __getitem__(self, key):
        return self.children.get(key,None)
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
        return "QNode(%.3f, %.3f | %s)" % (self.num_visits, self.value, str(self.children.keys()))
    def __repr__(self):
        return self.__str__()

cdef class VNode(TreeNode):
    def __init__(self, num_visits, value, **kwargs):
        self.num_visits = num_visits
        self.value = value
        self.children = {}  # a -> QNode
    def __str__(self):
        return "VNode(%.3f, %.3f | %s)" % (self.num_visits, self.value,
                                           str(self.children.keys()))
    def __repr__(self):
        return self.__str__()

    def print_children_value(self):
        for action in self.children:
            print("   action %s: %.3f" % (str(action), self[action].value))
    

cdef class RootVNode(VNode):
    def __init__(self, num_visits, value, history):
        VNode.__init__(self, num_visits, value)
        self.history = history
    @classmethod
    def from_vnode(cls, vnode, history):
        rootnode = RootVNode(vnode.num_visits, vnode.value, history)
        rootnode.children = vnode.children
        return rootnode

def print_tree_helper(root, parent_edge, depth, max_depth=None, complete=False):
    if max_depth is not None and depth >= max_depth:
        return
    print("%s%s" % ("    "*depth, str(parent_edge)))
    print("%s-%s" % ("    "*depth, str(root)))
    for c in root.children:
        if complete or (root[c].num_visits > 1):
            print_tree_helper(root[c], c, depth+1, max_depth=max_depth, complete=complete)

def print_tree(tree, max_depth=None, complete=False):
    print_tree_helper(tree, "", 0, max_depth=max_depth, complete=complete)

def print_preferred_actions(tree, max_depth=None):
    print_preferred_actions_helper(tree, 0, max_depth=max_depth)

def print_preferred_actions_helper(root, depth, max_depth=None):
    if max_depth is not None and depth >= max_depth:
        return
    best_child = None
    best_value = float('-inf')
    for c in root.children:
        if root[c].value > best_value:
            best_child = c
            best_value = root[c].value
    equally_good = []
    if isinstance(root, VNode):
        for c in root.children:
            if c != best_child and root[c].value == best_value:
                equally_good.append(c)

    if best_child is not None and root[best_child] is not None:
        if isinstance(root[best_child], QNode):
            print("  %s  %s" % (str(best_child), str(equally_good)))
        print_preferred_actions_helper(root[best_child], depth+1, max_depth=max_depth)            

cdef class ActionPrior:
    """A problem-specific object"""
    
    def get_preferred_actions(self, vnode, state=None, history=None,
                              belief=None, **kwargs):
        """
        This is to mimic the behavior of Simulator.Prior
        and GenerateLegal/GeneratePreferred in David Silver's
        POMCP code.

        Returns a set of preferred actions and associated
        num_visits_init and value_init, given arguments.
        In POMCP, this acts as a history-based prior policy,
        and in DESPOT, this acts as a belief-based prior policy.
        For example, given certain state or history, only it
        is possible that only a subset of all actions is legal;
        This is particularly true in the RockSample problem."""
        raise NotImplementedError
    
    def __init__(self, action, nvi, vi):
        self.action = action
        self.num_visits_init = nvi  # name it short to make cython compile work.
        self.value_init = vi
    
cdef class RolloutPolicy(PolicyModel):
    cpdef Action rollout(self, State state, tuple history=None):
        pass
    
cdef class RandomRollout(RolloutPolicy):
    cpdef Action rollout(self, State state, tuple history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]
    
cdef class POUCT(Planner):

    """POUCT only works for problems with action space
    that can be enumerated."""

    def __init__(self,
                 max_depth=5, planning_time=1.,
                 discount_factor=0.9, exploration_const=math.sqrt(2),
                 num_visits_init=1, value_init=0,
                 rollout_policy=RandomRollout(),
                 action_prior=None):
        """
        rollout_policy(vnode, state=?) -> a; default random rollout.
        action_prior (ActionPrior), see above.
        """
        self._max_depth = max_depth
        self._planning_time = planning_time
        self._num_visits_init = num_visits_init
        self._value_init = value_init
        self._rollout_policy = rollout_policy
        self._discount_factor = discount_factor
        self._exploration_const = exploration_const
        self._action_prior = action_prior

        # to simplify function calls; plan only for one agent at a time
        self._agent = None
        self._last_num_sims = -1

    @property
    def updates_agent_belief(self):
        return False

    @property
    def last_num_sims(self):
        """Returns the number of simulations ran for the last `plan` call."""
        return self._last_num_sims

    cpdef public plan(self, Agent agent):
        self._agent = agent   # switch focus on planning for the given agent
        if not hasattr(self._agent, "tree"):
            self._agent.add_attr("tree", None)
        action, num_sims = self._search()
        self._last_num_sims = num_sims
        return action            

    cpdef public update(self, Action real_action, Observation real_observation):
        """
        Assume that the agent's history has been updated after taking real_action
        and receiving real_observation.
        """
        if not hasattr(self._agent, "tree"):
            print("Warning: agent does not have tree. Have you planned yet?")
            return

        if self._agent.tree[real_action][real_observation] is not None:
            # Update the tree (prune)
            self._agent.tree = RootVNode.from_vnode(
                self._agent.tree[real_action][real_observation],
                self._agent.history)
        else:
            # observation was never encountered in simulation.
            self._agent.tree = None

    def clear_agent(self):
        self._agent = None  # forget about current agent so that can plan for another agent.
        self._last_num_sims = -1

    cpdef _expand_vnode(self, VNode vnode, tuple history, State state=None):
        cdef Action action    
        if self._action_prior is not None:
            # Using action prior; special values are set.
            for preference\
                in self._agent.policy_model.get_preferred_actions(vnode,
                                                                  state=state,
                                                                  history=history):
                action = preference.action
                if vnode[action] is None:
                    history_action_node = QNode(preference.num_visits_init,
                                                preference.value_init)
                    vnode[action] = history_action_node
        else:
            for action in self._agent.valid_actions(state=state, history=history):
                if vnode[action] is None:
                    history_action_node = QNode(self._num_visits_init,
                                                self._value_init)
                    vnode[action] = history_action_node

    def _sample_belief(self, agent):
        return agent.belief.random()

    cpdef _search(self):
        cdef State state
        cdef int num_sims = 0
        cdef Action best_action
        cdef float best_value
        
        start_time = time.time()
        while time.time() - start_time < self._planning_time:
            ## Note: the tree node with () history will have
            ## the init belief given to the agent.
            state = self._sample_belief(self._agent)
            self._simulate(state, self._agent.history, self._agent.tree,
                           None, None, 0)
            num_sims +=1
            
        best_action, best_value = None, float('-inf')            
        for action in self._agent.tree.children:
            if self._agent.tree[action].value > best_value:
                best_value = self._agent.tree[action].value
                best_action = action
        return best_action, num_sims

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
            # (for example, the state is not in the initiation set of the option)
            return reward

        total_reward = reward + (self._discount_factor**nsteps)*self._simulate(next_state,
                                                                               history + ((action, observation),),
                                                                               root[action][observation],
                                                                               root[action],
                                                                               observation,
                                                                               depth+nsteps)
        root.num_visits += 1
        root[action].num_visits += 1
        root.value = root.value + (total_reward - root.value) / (root.num_visits)
        root[action].value = root[action].value + (total_reward - root[action].value) / (root[action].num_visits)
        return total_reward

    cpdef _rollout(self, State state, tuple history, VNode root, int depth):
        cdef Action action
        cdef float discount = 1.0
        cdef float total_discounted_reward = 0
        cdef State next_state
        cdef Observation observation
        cdef float reward
        
        while depth <= self._max_depth:
            action = self._rollout_policy.rollout(state, history=history)
            next_state, observation, reward, nsteps = sample_generative_model(self._agent, state, action)
            if root[action] is None:
                history_action_node = QNode(self._num_visits_init, self._value_init)
                root[action] = history_action_node
            if observation not in root[action]:
                root[action][observation] = self._VNode()
                self._expand_vnode(root[action][observation], history, state=next_state)
            history = history + ((action, observation),)
            depth += nsteps
            total_discounted_reward += reward * discount
            discount *= (self._discount_factor**nsteps)
            state = next_state
            root = root[action][observation]
        return total_discounted_reward

    cpdef Action _ucb(self, VNode root):
        """UCB1"""
        cdef Action best_action
        cdef float best_value
        best_action, best_value = None, float('-inf')
        for action in root.children:
            val = root[action].value + \
                self._exploration_const * math.sqrt(math.log(root.num_visits) / root[action].num_visits)
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
            return RootVNode(self._num_visits_init,
                             self._value_init,
                             self._agent.history)
        else:
            return VNode(self._num_visits_init,
                         self._value_init)
    
