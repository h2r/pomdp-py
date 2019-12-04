class Environment:
    """An Environment maintains the true state of the world.
    For example, it is the 2D gridworld, rendered by pygame.
    Or it could be the 3D simulated world rendered by OpenGL.
    Therefore, when coding up an Environment, the developer
    should have in mind how to represent the state so that
    it can be used by a POMDP or OOPOMDP.

    The Environment is passive. It never observes nor acts.
    """
    def __init__(self, init_state,
                 transition_model,
                 reward_model):
        self._init_state = init_state
        self._transition_model = transition_model
        self._reward_model = reward_model
        self._cur_state = init_state

    @property
    def state(self):
        return self.cur_state

    @property
    def cur_state(self):
        return self._cur_state

    @property
    def transition_model(self):
        return self._transition_model

    @property
    def reward_model(self):
        return self._reward_model
    
    def state_transition(self, action, execute=True, **kwargs):
        """Modifies current state of the environment"""
        next_state = self.transition_model.sample(self.state, action, **kwargs)
        if execute:
            self._cur_state = next_state

    def provide_observation(self, observation_model, action, **kwargs):
        return observation_model.sample(self.state, action, **kwargs)

    
    
