"""
Some particular implementations of the interface for convenience
"""
import pomdp_py
import random

class SimpleState(pomdp_py.State):
    """A SimpleState is a state that stores
    one piece of hashable data and the equality
    of two states of this kind depends just on
    this data"""
    def __init__(self, data):
        self.data = data
    def __hash__(self):
        return hash(self.data)
    def __eq__(self, other):
        if isinstance(other, SimpleState):
            return self.data == other.data
        return False
    def __str__(self):
        return str(self.data)
    def __repr__(self):
        return "SimpleState({})".format(self.data)

class SimpleAction(pomdp_py.Action):
    """A SimpleAction is an action defined by a string name"""
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, SimpleAction):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "SimpleAction(%s)" % self.name

class SimpleObservation(pomdp_py.Observation):
    """A SimpleObservation is an observation
    with a piece of hashable data that defines
    the equality."""
    def __init__(self, data):
        self.data = data
    def __hash__(self):
        return hash(self.data)
    def __eq__(self, other):
        if isinstance(other, SimpleObservation):
            return self.data == other.data
        return False
    def __str__(self):
        return str(self.data)
    def __repr__(self):
        return "SimpleObservation(%s)" % self.data

class DetTransitionModel(pomdp_py.TransitionModel):
    """A DetTransitionModel is a deterministic transition model.
    A probability of 1 - epsilon is given for correct transition,
    and epsilon is given for incorrect transition."""
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon

    def probability(self, next_state, state, action):
        """According to problem spec, the world resets once
        action is open-left/open-right. Otherwise, stays the same"""
        if self.sample(state, action) == next_state:
            return 1.0 - self.epsilon
        else:
            return self.epsilon

    def sample(self, state, action):
        raise NotImplementedError


class DetObservationModel(pomdp_py.ObservationModel):
    """A DetTransitionModel is a deterministic transition model.
    A probability of 1 - epsilon is given for correct transition,
    and epsilon is given for incorrect transition."""
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon

    def probability(self, observation, next_state, action):
        """According to problem spec, the world resets once
        action is open-left/open-right. Otherwise, stays the same"""
        if self.sample(next_state, action) == observation:
            return 1.0 - self.epsilon
        else:
            return self.epsilon

    def sample(self, next_state, action):
        raise NotImplementedError


class DetRewardModel(pomdp_py.RewardModel):
    """A DetRewardModel is a deterministic reward model (the most typical kind)."""
    def reward_func(self, state, action, next_state):
        raise NotImplementedError

    def sample(self, state, action, next_state):
        # deterministic
        return self.reward_func(state, action, next_state)

    def argmax(self, state, action, next_state):
        return self.sample(state, action, next_state)

class UniformPolicyModel(pomdp_py.RolloutPolicy):
    def __init__(self, actions):
        self.actions = actions

    def sample(self, state, **kwargs):
        return random.sample(self.actions, 1)[0]

    def get_all_actions(self, state=None, history=None):
        return self.actions

    def rollout(self, state, history=None):
        return random.sample(self.actions, 1)[0]
