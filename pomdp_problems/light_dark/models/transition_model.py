"""Defines the TransitionModel for the continuous light-dark domain;

Origin: Belief space planning assuming maximum likelihood observations

Quote from the paper:

    The underlying system dynamics are linear with zero process noise,
    :math:`f(x_t,u_t)=x_t+u`. This means the transition dynamics is
    deterministic.
"""
import pomdp_py
import copy
import numpy as np


class TransitionModel(pomdp_py.TransitionModel):
    """
    The underlying deterministic system dynamics
    """
    def __init__(self, epsilon=1e-9):
        self._epsilon = epsilon

    def probability(self, next_state, state, action, **kwargs):
        """
        Deterministic.
        """
        expected_position = tuple(self.func(state.position, action))
        if next_state.position == expected_position:
            return 1.0 - self.epsilon
        else:
            return self.epsilon

    def sample(self, state, action):
        next_state = copy.deepcopy(state)
        next_state.position = tuple(self.func(state.position, action))
        return next_state

    def argmax(self, state, action):
        """Returns the most likely next state"""
        return self.sample(state, action)

    def func(self):
        """Returns the function of the underlying system dynamics.
        The function is: (xt, ut) -> xt+1 where xt, ut, xt+1 are
        all numpy arrays."""
        def f(xt, ut):
            return np.array([xt[0] + ut[0],
                             xt[1] + ut[1]])
        return f

    def jac_dx(self):
        """Returns the function of the jacobian of the system dynamics
        function with respect to the state vector mt: (mt, ut) -> At"""
        def dfdx(mt, ut):
            # The result of computing the jacobian by hand
            return np.array([[ut[0], mt[1] + ut[1]],
                             [mt[0] + ut[0], ut[1]]])
        return dfdx

    def jac_du(self):
        """Returns the function of the jacobian of the system dynamics
        function with respect to the state vector mt: (mt, ut) -> Bt"""
        def dfdu(mt, ut):
            # The result of computing the jacobian by hand
            return np.array([[mt[0], mt[1] + ut[1]],
                             [mt[0] + ut[0], mt[1]]])
        return dfdu
    
    def func_noise(self, var_sysd=1e-9):
        """Returns a function that returns a state-dependent Gaussian noise."""
        def fn(mt):
            gaussian_noise = pomdp_py.Gaussian([0,0],
                                               [[var_sysd, 0],
                                                [0, var_sysd]])
            return gaussian_noise
        return fn
