# Implementation of the basic policy tree based
# value iteration as explained in section 4.1 of
# Planning and acting in partially observable stochastic domains
# https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf
#
# Warning: No pruning - the number of policy trees explodes very fast.

from pomdp_py.framework.planner cimport Planner
from pomdp_py.framework.basics cimport Agent, Action, State
import numpy as np
import itertools

cdef class PolicyTreeNode:

    cdef public Action action
    cdef public int depth
    cdef public dict children, values
    cdef Agent _agent
    cdef float _discount_factor
    
    def __init__(self, action, depth, agent, discount_factor, children={}):
        self.action = action
        self.depth = depth
        self._agent = agent
        self.children = children
        self._discount_factor = discount_factor
        self.values = self._compute_values()  # s -> value
        
    def _compute_values(self):
        """
        Returns a dictionary {s -> value} that represents the values
        for the next actions.
        """
        actions = self._agent.all_actions
        observations = self._agent.all_observations
        states = self._agent.all_states
        
        discount_factor = self._discount_factor**self.depth
        values = {}
        for s in states:
            expected_future_value = 0.0
            for sp in states:
                for o in observations:
                    trans_prob = self._agent.transition_model.probability(sp, s, self.action)
                    obsrv_prob = self._agent.observation_model.probability(o, sp, self.action)
                    if len(self.children) > 0:
                        subtree_value = self.children[o].values[s]  # corresponds to V_{oi(p)} in paper
                    else:
                        subtree_value = 0.0
                    reward = self._agent.reward_model.sample(s, self.action, sp)
                    expected_future_value += trans_prob * obsrv_prob * (reward + discount_factor*subtree_value)
            values[s] = expected_future_value
        return values

    def __str__(self):
        return "PolicyTreeNode(%s, %d){%s}" % (self.action, self.depth, str(list(self.children.keys())))
    def __repr__(self):
        return self.__str__()
        

cdef class ValueIteration(Planner):
    """
    This algorithm is only feasible for small problems where states, actions,
    and observations can be explicitly enumerated.
    """
    cdef float _discount_factor, _epsilon
    cdef int _planning_horizon    

    def __init__(self, horizon=float('inf'), discount_factor=0.9, epsilon=1e-6):
        """The horizon satisfies discount_factor**horizon > epsilon"""
        self._discount_factor = discount_factor
        self._epsilon = epsilon
        self._planning_horizon = horizon

    def plan(self, agent):
        policy_trees = self._build_policy_trees(0, agent)
        value_beliefs = {}
        for p, policy_tree in enumerate(policy_trees):
            value_beliefs[p] = 0
            for state in agent.all_states:
                value_beliefs[p] += agent.cur_belief[state] * policy_tree.values[state]
        # Pick the policy tree with highest belief value
        pmax = max(value_beliefs, key=value_beliefs.get)
        return policy_trees[pmax].action

    def _nodes_by_level(self, node, depth, levelmap):
        if depth not in levelmap:
            levelmap[depth] = []
        levelmap[depth].append(node)
        for o in sorted(node.children):
            self._nodes_by_level(node.children[o], depth+1, levelmap)
            
    def _value_policy_tree(self, policy_tree_root):
        levelmap = {}
        self._nodes_by_level(policy_tree_root, 0, levelmap)
        horizon = max(levelmap.keys())

        while horizon >= 1:
            subtrees = levelmap[horizon]

    def _build_policy_trees(self, depth, agent):
        """Bottom up build policy trees"""
        actions = agent.all_actions
        observations = agent.all_observations
        states = agent.all_states

        if depth >= self._planning_horizon or self._discount_factor**depth < self._epsilon:
            return [PolicyTreeNode(a, depth, agent, self._discount_factor) for a in actions]
        else:
            # Every observation can lead to K possible sub policy trees,
            # which is exactly the output of _build_policy_trees. Then,
            # for every set of |O| observations, there is a permutation
            # of sub policy trees. In other words, every permutation of
            # |O| sub policy trees corresponds to the children of a node
            # (root, i.e. action) in the policy tree. Return all possible
            # roots of such. The permutations are result of cross product
            # between the subtrees for different observations.
            groups = {}
            group_size = 0
            for o in observations:
                groups[o] = self._build_policy_trees(depth+1, agent)
                if group_size > 0:
                    assert len(groups[o]) == group_size
                group_size = len(groups[o])
                
            permutations = itertools.product(*([np.arange(group_size)]*len(observations)))
            policy_trees = []
            for perm in permutations:
                for i in perm:
                    children = {o:groups[o][i]
                                for o in observations}
                    for a in actions:
                        policy_tree_node = PolicyTreeNode(a, depth, agent, self._discount_factor)
                        policy_tree_node.children = children
                        policy_trees.append(policy_tree_node)
            return policy_trees
