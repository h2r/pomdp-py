"""Defines the TransitionModel for the continuous light-dark domain;

Origin: Belief space planning assuming maximum likelihood observations

Quote from the paper:

    The underlying system dynamics are linear with zero process noise,
    :math:`f(x_t,u_t)=x_t+u`. This means the transition dynamics is
    deterministic.

"""
import pomdp_py
import copy

class LightDarkTransitionModel(pomdp_py.TransitionModel):

    def __init__(self, epsilon=1e-9):
        self._epsilon = epsilon

    def probability(self, next_state, state, action, **kwargs):
        """
        Deterministic.
        """
        expected_position = (state.position[0] + action.control[0],
                             state.position[1] + action.control[1])
        if next_state.position == expected_position:
            return 1.0 - self.epsilon
        else:
            return self.epsilon

    def sample(self, state, action):
        next_state = copy.deepcopy(state)
        next_state.position = (state.position[0] + action.control[0],
                               state.position[1] + action.control[1])
        return next_state

    def argmax(self, state, action):
        """Returns the most likely next state"""
        return self.sample(state, action)
