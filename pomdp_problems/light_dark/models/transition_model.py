"""Defines the TransitionModel for the continuous light-dark domain;

Origin: Belief space planning assuming maximum likelihood observations

Quote from the paper:

    The underlying system dynamics are linear with zero process noise,
    :math:`f(x_t,u_t)=x_t+u`. This means the transition dynamics is
    deterministic.

Also, includes BeliefSpaceTransitionModel. Basically the "belief space dynamics",
which is a main part of the original paper.
"""
import pomdp_py
import copy


class TransitionModel(pomdp_py.TransitionModel):

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


class BeliefSpaceTransitionModel(pomdp_py.TransitionModel):
    """
    This is the Belief Space Dynamics Model; It is a TransitionModel
    but the states are BeliefState(s).
    
    Refer to Section III. Simplified Belief Space Dynamics of the paper.
    """
    def __init__(self):
        pass

    def probability(self, next_state, state, action, **kwargs):
        pass

    def sample(self, state, action):
        pass





                    
