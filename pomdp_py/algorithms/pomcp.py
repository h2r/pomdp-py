# We implement POMCP as described in the original paper
# Monte-Carlo Planning in Large POMDPs
# https://papers.nips.cc/paper/4031-monte-carlo-planning-in-large-pomdps


# One thing to note is that, in this algorithm, belief
# update happens as the simulation progresses. The new
# belief is stored in the vnodes at the level after
# executing the next action. These particles will
# be reinvigorated if they are not enough.
#     However, it is possible to separate MCTS completely
# from the belief update. This means the belief nodes
# no longer keep track of particles, and belief update
# and particle reinvogration happen for once after MCTS
# is completed. I have previously implemented this version.
# This version is also implemented in BasicPOMCP.jl
# (https://github.com/JuliaPOMDP/BasicPOMCP.jl)
# The two should be EQUIVALENT. In general, it doesn't
# hurt to do the belief update during MCTS, a feature
# of using particle representation.

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
        return "QNode(%.3f, %.3f | %s)->%s" % (self.num_visits, self.value, str(self.children.keys()))
    def __repr__(self):
        return self.__str__()

class VNode(TreeNode):
    def __init__(self, num_visits, value, belief):
        self.num_visits = num_visits
        self.value = value
        self.belief = belief
        self.children = {}  # a -> QNode
    def __str__(self):
        return "VNode(%.3f, %.3f, %d | %s)" % (self.num_visits, self.value, len(self.belief),
                                               str(self.children.keys()))
    def __repr__(self):
        return self.__str__()

class RootVNode(VNode):
    def __init__(self, num_visits, value, belief, history):
        VNode.__init__(self, num_visits, value, belief)
        self.history = history
    @classmethod
    def from_vnode(cls, vnode, history):
        rootnode = RootVNode(vnode.num_visits, vnode.value, vnode.belief, history)
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

class POMCP(Planner):

    """This POMCP version only works for problems
    with action space that can be enumerated."""

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
    def update_agent_belief(self):
        """True if planner's update function also updates agent's
        belief."""
        return True

    def plan(self, agent, action_prior_args={}):        
        # Only works if the agent's belief is particles
        if not isinstance(agent.belief, Particles):
            raise TypeError("Agent's belief is not represented in particles.\n"\
                            "POMCP not usable. Please convert it to particles.")
        self._agent = agent   # switch focus on planning for the given agent
        if not hasattr(self._agent, "tree"):
            self._agent.add_attr("tree", None)
        
        return self._search(action_prior_args=action_prior_args)

    def update(self, agent, real_action, real_observation, action_prior_args={},
               state_transform_func=None, ):
        """
        Assume that the agent's history has been updated after taking real_action
        and receiving real_observation.

        `state_transform_func`: Used to add artificial transform to states during
            particle reinvigoration. Signature: s -> s_transformed
        """
        if not isinstance(agent.belief, Particles):
            raise TypeError("Agent's belief is not represented in particles.\n"\
                            "POMCP not usable. Please convert it to particles.")
        if not hasattr(self._agent, "tree"):
            print("Warning: agent does not have tree. Have you planned yet?")
            return
        # Update the tree; Reinvigorate the tree's belief and use it
        # as the updated belief for the agent.
        self._agent.tree = RootVNode.from_vnode(self._agent.tree[real_action][real_observation],
                                                self._agent.history)
        if self._agent.tree is None:
            # Never anticipated the real_observation. No reinvigoration can happen.
            raise ValueError("Particle deprivation.")
        tree_belief = self._agent.tree.belief
        self._agent.set_belief(self._particle_reinvigoration(tree_belief,
                                                             real_action,
                                                             real_observation,
                                                             len(self._agent.init_belief.particles),
                                                             state_transform_func=state_transform_func))
        if self._agent.tree is None:
            # observation was never encountered in simulation.
            self._agent.tree = RootVNode(self._num_visits_init,
                                         self._value_init,
                                         copy.deepcopy(agent.belief),
                                         self._agent.history)
            self._expand_vnode(self._agent.tree, self._agent.history,
                               action_prior_args=action_prior_args)
        else:
            self._agent.tree.belief = copy.deepcopy(self._agent.belief)

    def _particle_reinvigoration(self, particles, real_action,
                                 real_observation, num_particles, state_transform_func=None):
        """Note that particles should contain states that have already made
        the transition as a result of the real action. Therefore, they simply
        form part of the reinvigorated particles. At least maintain `num_particles`
        number of particles. If already have more, then it's ok.
        """
        # If not enough particles, introduce artificial noise to existing particles (reinvigoration)
        new_particles = copy.deepcopy(particles)
        if len(new_particles) == 0:
            raise ValueError("Particle deprivation.")

        if len(new_particles) > num_particles:
            return new_particles
        
        print("Particle reinvigoration for %d particles" % (num_particles - len(new_particles)))
        while len(new_particles) < num_particles:
            # need to make a copy otherwise the transform affects states in 'particles'
            next_state = copy.deepcopy(particles.random())
            # Add artificial noise
            if state_transform_func is not None:
                next_state = state_transform_func(next_state)
            new_particles.add(next_state)
        return new_particles

    def _expand_vnode(self, vnode, history, state=None, action_prior_args={}):
        if self._action_prior is not None:
            # Using action prior; special values are set.
            for action, num_vists_init, value_init\
                in self._agent.policy_model.get_preferred_actions(state=state,
                                                                  history=history,
                                                                  belief=vnode.belief,
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

    def _search(self, action_prior_args={}):
        # Initialize the tree, if previously empty.
        if self._agent.tree is None:
            self._agent.tree = RootVNode(self._num_visits_init,
                                         self._value_init,
                                         copy.deepcopy(self._agent.belief),
                                         self._agent.history)
            self._expand_vnode(self._agent.tree, self._agent.history,
                               action_prior_args=action_prior_args)
        # Locate the tree node with given history
        result = self._verify_history(self._agent.history)
        if not result:
            raise ValueError("Unable to plan for the given history.")
        tree, parent = result

        start_time = time.time()
        while time.time() - start_time < self._planning_time:
            ## Note: the tree node with () history will have
            ## the init belief given to the agent.
            state = tree.belief.random()
            self._simulate(state, self._agent.history, tree, parent, None, 0,
                           action_prior_args=action_prior_args)
            
        best_action, best_value = None, float('-inf')            
        for action in tree.children:
            if tree[action].value > best_value:
                best_value = tree[action].value
                best_action = action
            # print("action %s: %.3f" % (str(action), tree[action].value))
        return best_action

    def _verify_history(self, history):
        """Returns corresponding node if the given history can be used
        for search. Else None."""
        parent = None
        tree = self._agent.tree
        if len(history) < len(tree.history):
            # history too early. Real action already taken
            return None
        else:
            common_history = history[:len(tree.history)]
            if common_history != history:
                """Histories are different."""
                return None
            else:
                extra_history = history[len(tree.history):]
                current_node = tree
                for action, observation in extra_history:
                    if action not in current_node.children:
                        # history not encountered
                        return None
                    else:
                        qnode = current_node.children[action]
                        if observation not in qnode:
                            # history not encountered
                            return None
                        current_node = qnode.children[observation]
                        parent = qnode
                return current_node, parent

    def _simulate(self, state, history, root, parent, observation, depth, action_prior_args={}):
        if depth > self._max_depth:
            return 0
        if root is None:
            root = VNode(self._num_visits_init,
                         self._value_init,
                         Particles([]))
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
        if depth == 1:
            root.belief.add(state)  # belief update happens as simulation goes.
        root.num_visits += 1
        root[action].num_visits += 1
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
            root[action][observation] = VNode(self._num_visits_init,
                                              self._value_init,
                                              Particles([]))
            self._expand_vnode(root[action][observation], history, action_prior_args=action_prior_args)
        return reward + self._discount_factor * self._rollout(next_state,
                                                              history + ((action, observation)),
                                                              root[action][observation],
                                                              depth+1)

    def _ucb(self, root):
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


        
    
                
