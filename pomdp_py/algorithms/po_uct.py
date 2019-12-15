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

from abc import ABC, abstractmethod 
from pomdp_py.framework.planner import Planner
from pomdp_py.representations.distribution.particles import Particles
import copy
import time
import random
import math

class TreeNode:
    def __init__(self):
        self.children = {}
    def __getitem__(self, key):
        return self.children.get(key,None)
    def __setitem__(self, key, value):
        self.children[key] = value
    def __contains__(self, key):
        return key in self.children
    def __hash__(self):
        return hash(id(self))
    def __eq__(self, other):
        return id(self) == id(other)

class QNode(TreeNode):
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

class VNode(TreeNode):
    def __init__(self, num_visits, value, **kwargs):
        self.num_visits = num_visits
        self.value = value
        self.children = {}  # a -> QNode
    def __str__(self):
        return "VNode(%.3f, %.3f | %s)" % (self.num_visits, self.value,
                                           str(self.children.keys()))
    def __repr__(self):
        return self.__str__()

class RootVNode(VNode):
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
    for c in root.children:
        if c != best_child and root[c].value == best_value:
            equally_good.append(c)

    if best_child is not None and root[best_child] is not None:
        if isinstance(root[best_child], QNode):
            print("  %s  %s" % (str(best_child), str(equally_good)))
        print_preferred_actions_helper(root[best_child], depth+1, max_depth=max_depth)            

class ActionPrior(ABC):
    """A problem-specific object"""
    @abstractmethod
    def get_preferred_actions(self, state=None, history=None,
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
        raise NotImplemented

def random_rollout(vnode, state=None):
    return random.choice(list(vnode.children))

class POUCT(Planner):

    """POUCT only works for problems with action space
    that can be enumerated."""

    def __init__(self,
                 max_depth=5, planning_time=1.,
                 discount_factor=0.9, exploration_const=math.sqrt(2),
                 num_visits_init=1, value_init=0,
                 rollout_policy=random_rollout,
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

    @property
    def updates_agent_belief(self):
        return False

    def plan(self, agent, action_prior_args={}):
        self._agent = agent   # switch focus on planning for the given agent
        if not hasattr(self._agent, "tree"):
            self._agent.add_attr("tree", None)

        action = self._search(action_prior_args=action_prior_args)
        self._agent = None  # forget about current agent so that can plan for another agent.
        return action            

    def update(self, agent, real_action, real_observation, action_prior_args={}, **kwargs):
        """
        Assume that the agent's history has been updated after taking real_action
        and receiving real_observation.
        """
        if not hasattr(agent, "tree"):
            print("Warning: agent does not have tree. Have you planned yet?")
            return

        if agent.tree[real_action][real_observation] is not None:
            # Update the tree (prune)
            agent.tree = RootVNode.from_vnode(agent.tree[real_action][real_observation],
                                              agent.history)
        else:
            # observation was never encountered in simulation.
            agent.tree = RootVNode(self._num_visits_init,
                                   self._value_init,
                                   agent.history)
            self._expand_vnode(agent.tree, agent.history,
                               action_prior_args=action_prior_args)

    def _expand_vnode(self, vnode, history, state=None, action_prior_args={}):
        if self._action_prior is not None:
            # Using action prior; special values are set.
            for action, num_vists_init, value_init\
                in self._agent.policy_model.get_preferred_actions(state=state,
                                                                  history=history,
                                                                  **action_prior_args):
                if vnode[action] is None:
                    history_action_node = QNode(num_visits_init,
                                                value_init)
                    vnode[action] = history_action_node
        else:
            for action in self._agent.all_actions:
                if vnode[action] is None:
                    history_action_node = QNode(self._num_visits_init,
                                                self._value_init)
                    vnode[action] = history_action_node

    def _sample_belief(self, agent):
        return agent.belief.random()

    def _search(self, action_prior_args={}):
        # Initialize the tree, if previously empty.
        if self._agent.tree is None:
            self._agent.tree = self._VNode(agent=self._agent, root=True)
            self._expand_vnode(self._agent.tree, self._agent.history,
                               action_prior_args=action_prior_args)
        # Verify history
        if self._agent.tree.history != self._agent.history:
            raise ValueError("Unable to plan for the given history.")

        start_time = time.time()
        while time.time() - start_time < self._planning_time:
            ## Note: the tree node with () history will have
            ## the init belief given to the agent.
            state = self._sample_belief(self._agent)
            self._simulate(state, self._agent.history, self._agent.tree,
                           None, None, 0, action_prior_args=action_prior_args)
            
        best_action, best_value = None, float('-inf')            
        for action in self._agent.tree.children:
            if self._agent.tree[action].value > best_value:
                best_value = self._agent.tree[action].value
                best_action = action
            # print("action %s: %.3f" % (str(action), tree[action].value))
        return best_action

    def _simulate(self, state, history, root, parent, observation, depth, action_prior_args={}):
        if depth > self._max_depth:
            return 0
        if root is None:
            root = self._VNode()
            if parent is not None:
                parent[observation] = root            
            self._expand_vnode(root, history, state=state, action_prior_args=action_prior_args)
            rollout_reward = self._rollout(state, history, root, depth)
            return rollout_reward
        action = self._ucb(root)
        next_state, observation, reward = self._sample_generative_model(state, action)
        total_reward = reward + self._discount_factor*self._simulate(next_state,
                                                                     history + ((action, observation)),
                                                                     root[action][observation],
                                                                     root[action],
                                                                     observation,
                                                                     depth+1)
        root.num_visits += 1
        root[action].num_visits += 1
        root.value = root.value + (total_reward - root.value) / (root.num_visits)
        root[action].value = root[action].value + (total_reward - root[action].value) / (root[action].num_visits)
        return total_reward

    def _rollout(self, state, history, root, depth, action_prior_args={}):
        if depth > self._max_depth:
            return 0
        action = self._rollout_policy(root, state=state)
        next_state, observation, reward = self._sample_generative_model(state, action)
        if root[action] is None:
            history_action_node = QNode(action, self._num_visits_init, self._value_init)
            root[action] = history_action_node
        if observation not in root[action]:
            root[action][observation] = self._VNode()
            self._expand_vnode(root[action][observation], history, action_prior_args=action_prior_args)
        return reward + self._discount_factor * self._rollout(next_state,
                                                              history + ((action, observation)),
                                                              root[action][observation],
                                                              depth+1)

    def _ucb(self, root):
        """UCB1"""
        best_action, best_value = None, float('-inf')
        for action in root.children:
            val = root[action].value + \
                self._exploration_const * math.sqrt(math.log(root.num_visits) / root[action].num_visits)
            if val > best_value:
                best_action = action
                best_value = val
        return best_action

    def _sample_generative_model(self, state, action):
        '''
        (s', o, r) ~ G(s, a)
        '''
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
    
