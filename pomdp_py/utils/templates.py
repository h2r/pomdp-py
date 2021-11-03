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
    def __ne__(self, other):
        return not self.__eq__(other)
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
    def __ne__(self, other):
        return not self.__eq__(other)
    def __str__(self):
        return self.name
    def __repr__(self):
        return "SimpleAction({})".format(self.name)

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
    def __ne__(self, other):
        return not self.__eq__(other)
    def __str__(self):
        return str(self.data)
    def __repr__(self):
        return "SimpleObservation({})".format(self.data)

class DetTransitionModel(pomdp_py.TransitionModel):
    """A DetTransitionModel is a deterministic transition model.
    A probability of 1 - epsilon is given for correct transition,
    and epsilon is given for incorrect transition."""
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon

    def probability(self, next_state, state, action):
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


# Tabular models
class TabularTransitionModel(pomdp_py.TransitionModel):
    """This tabular transition model is built given a dictionary that maps a tuple
    (state, action, next_state) to a probability.  This model assumes that the
    given `weights` is complete, that is, it specifies the probability of all
    state-action-nextstate combinations
    """
    def __init__(self, weights):
        self.weights = weights
        self._states = set()
        for s, _, sp in weights:
            self._states.add(s)
            self._states.add(sp)

    def probability(self, next_state, state, action):
        if (state, action, next_state) in self.weights:
            return self.weights[(state, action, next_state)]
        raise ValueError("The transition probability for"\
                         f"{(state, action, next_state)} is not defined")

    def sample(self, state, action):
        next_states = list(self._states)
        probs = [self.probability(next_state, state, action)
                 for next_state in next_states]
        return random.choices(next_states, weights=probs, k=1)[0]

    def get_all_states(self):
        return self._states


class TabularObservationModel(pomdp_py.ObservationModel):
    """This tabular observation model is built given a dictionary that maps a tuple
    (next_state, action, observation) to a probability.  This model assumes that the
    given `weights` is complete.
    """
    def __init__(self, weights):
        self.weights = weights
        self._observations = set()
        for _, _, z in weights:
            self._observations.add(z)

    def probability(self, observation, next_state, action):
        """observation is emitted from state"""
        if (next_state, action, observation) in self.weights:
            return self.weights[(next_state, action, observation)]
        elif (next_state, observation) in self.weights:
            return self.weights[(next_state, observation)]
        raise ValueError("The observation probability for"
                         f"{(next_state, action, observation)} or {(next_state, observation)}"
                         "is not defined")

    def sample(self, next_state, action):
        observations = list(self._observations)
        probs = [self.probability(observation, next_state, action)
                 for observation in observations]
        return random.choices(observations, weights=probs, k=1)[0]

    def get_all_observations(self):
        return self._observations


class TabularRewardModel(pomdp_py.RewardModel):
    """This tabular reward model is built given a dictionary that maps a state or a
    tuple (state, action), or (state, action, next_state) to a probability.  This
    model assumes that the given `rewards` is complete.
    """
    def __init__(self, rewards):
        self.rewards = rewards

    def sample(self, state, action, *args):
        if state in self.rewards:
            return self.rewards[state]
        elif (state, action) in self.rewards:
            return self.rewards[(state, action)]
        else:
            if len(args) > 0:
                next_state = args[0]
                if (state, action, next_state) in self.rewards:
                    return self.rewards[(state, action, next_state)]

            raise ValueError("The reward is undefined for"
                             f"state={state}, action={action}"
                             f"next_state={args}")
