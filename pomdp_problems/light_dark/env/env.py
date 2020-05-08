"""Defines the Environment for the light dark domain.

Origin: Belief space planning assuming maximum likelihood observations
"""

import pomdp_py
import pomdp_problems.light_dark as ld
import numpy as np

class LightDarkEnvironment(pomdp_py.Environment):
    
    def __init__(self,
                 init_state,
                 light,
                 const,
                 reward_model=None):
        """
        Args:
            init_state (light_dark.domain.State or np.ndarray):
                initial true state of the light-dark domain,
            goal_pos (tuple): goal position (x,y)
            light (float):  see below
            const (float): see below
            reward_model (pomdp_py.RewardModel): A reward model used to evaluate a policy
        `light` and `const` are parameters in
        :math:`w(x) = \frac{1}{2}(\text{light}-s_x)^2 + \text{const}`

        Basically, there is "light" at the x location at `light`,
        and the farther you are from it, the darker it is.
        """
        self._light = light
        self._const = const
        transition_model = ld.TransitionModel()
        if type(init_state) == np.ndarray:
            init_state = ld.State(init_state)
        super().__init__(init_state,
                         transition_model,
                         reward_model)

    @property
    def light(self):
        return self._light

    @property
    def const(self):
        return self._const
