"""Defines the ObservationModel for the continuous light-dark domain;

Origin: Belief space planning assuming maximum likelihood observations

Quote from the paper:

    The observation function is identity, :math:`g(x_t) = x_t+\omega`,
    with zero-mean Gaussian observation noise a function of state,
    \omega\sim\mathcal{N}(\cdot | 0, w(x))` where

    :math:`w(x) = \frac{1}{2}(5-s_x)^2 + \text{const}`

    (Notational change; using :math:`s_x` to refer to first element of
    state (i.e. robot position). The number 5 indicates the x-coordinate
    of the light bar as shown in the figure (Fig.1 of the paper).
"""
import pomdp_py
import copy
from ..domain.observation import *

class LightDarkObservationModel(pomdp_py.ObservationModel):

    def __init__(self, light, const):
        """
        `light` and `const` are parameters in
        :math:`w(x) = \frac{1}{2}(\text{light}-s_x)^2 + \text{const}`

        They should both be floats. The quantity :math:`w(x)` will
        be used as the variance of the covariance matrix in the gaussian
        distribution (this is how I understood the paper).
        """
        self._light = light
        self._const = const

    def probability(self, observation, next_state, action):
        """
        The observation is :math:`g(x_t) = x_t+\omega`. So
        the probability of this observation is the probability
        of :math:`\omega` which follows the Gaussian distribution.
        """        
        variance = 0.5 * (self._light - next_state.position[0])**2 + self._const
        gaussian_noise = pomdp_py.Gaussian([0,0],
                                           [[variance, 0],
                                            [0, variance]])
        omega = (observation.position[0] - next_state.position[0],
                 observation.position[1] - next_state.position[1])
        return gaussian_noise[omega]

    def sample(self, next_state, action, argmax=False):
        """sample an observation."""
        # Sample a position shift according to the gaussian noise.
        variance = 0.5 * (self._light - next_state.position[0])**2 + self._const
        gaussian_noise = pomdp_py.Gaussian([0,0],
                                           [[variance, 0],
                                            [0, variance]])
        if mpe:
            omega = gaussian_noise.mpe()
        else:
            omega = gaussian_noise.random()
        return Observation((next_state.position[0] + omega[0],
                            next_state.position[1] + omega[1]))
        
    def argmax(self, next_state, action):
        return self.sample(next_state, action, argmax=True)
