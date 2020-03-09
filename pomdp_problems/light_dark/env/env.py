import pomdp_py
import pomdp_problems.light_dark as ld

class LightDarkEnvironment(pomdp_py.Environment):
    
    def __init__(self,
                 init_state, goal_pos, light,
                 const, discrete=False,
                 goal_tolerance=0.5,
                 minimum_motion=1e-3):
        """
        Args:
            init_state (light_dark.domain.State): initial true state of the light-dark domain,
            goal_pos (tuple): goal position (x,y)
            light (float)  see below
            const (float): see below
            goal_tolerance (float): goal tolerance radius
            motion_threshold (float): upper bound of the change of the robot position that
               can still be regarded as "staying"
    
        `light` and `const` are parameters in
        :math:`w(x) = \frac{1}{2}(\text{light}-s_x)^2 + \text{const}`

        Basically, there is "light" at the x location at `light`,
        and the farther you are from it, the darker it is.
        """
        self._light = light
        self._const = const
        self._goal_tolerance = goal_tolerance
        transition_model = ld.TransitionModel()
        if discrete:
            # sparse reward
            reward_model = ld.SparseRewardModel(goal_pos,
                                                tolerance=goal_tolerance,
                                                minimum_motion=minimum_motion)
        else:
            raise Value("Not yet implemented in this case.")
        super().__init__(init_state,
                         transition_model,
                         reward_model)

    @property
    def goal_pos(self):
        return self.reward_model.goal_pos

    @property
    def goal_tolerance(self):
        return self._goal_tolerance

    @property
    def light(self):
        return self._light

    @property
    def const(self):
        return self._const
