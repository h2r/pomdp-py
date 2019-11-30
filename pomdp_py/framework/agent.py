class Agent:

    def __init__(self, pomdp, init_belief,
                 policy_model,
                 transition_model=None,
                 observation_model=None,
                 reward_model=None,
                 blackbox_model=None):
        self._pomdp = pomdp
        self._init_belief = init_belief
        self._policy_model = policy_model
        
        self._transition_model = transition_model
        self._observation_model = observation_model
        self._reward_model = reward_model
        self._blackbox_model = blackbox_model
        # It cannot be the case that both explicit models and blackbox model are None.
        if self._blackbox_model is None:
            assert self._transition_model is not None\
                and self._observation_model is not None\
                and self._reward_model is not None

        # For online planning
        self._cur_belief = init_belief
        self._history = ()

    @property
    def belief(self):
        return self.cur_belief

    @property
    def cur_belief(self):
        return self._cur_belief

    @property
    def pomdp(self):
        return self._pomdp
    
    @property
    def observation_model(self):
        return self._observation_model

    @property
    def transition_model(self):
        return self._transition_model

    @property
    def reward_model(self):
        return self._reward_model

    @property
    def blackbox_model(self):
        return self._blackbox_model

    @property
    def generative_model(self):
        return self.blackbox_model

    def update(self, real_action, real_observation, **kwargs):
        """updates the history and performs belief update"""
        raise NotImplemented
    
