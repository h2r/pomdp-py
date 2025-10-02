import numpy.random 
import pomdp_py 

# ---------------------------------------------------------------
# BAMDP State 
# ---------------------------------------------------------------
class BAMDPState(pomdp_py.State):
    def __init__(self, name, counts):
        self.name = name 
        self.counts = counts    # dict {action_name: (success_count, failure_count)}

    def __hash__(self):
        return hash((self.name, frozenset(self.counts.items())))

    def __eq__(self, other):
        if isinstance(other, BAMDPState):
            return (self.name == other.name) and (self.counts == other.counts)
        return False

    def __str__(self):
        return f"Env: {self.name}, Counts: {self.counts}"

# ---------------------------------------------------------------
# BAMDP Action
# ---------------------------------------------------------------
class BAMDPAction(pomdp_py.Action):
    def __init__(self, name):
        self.name 

    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, BAMDPAction):
            return self.name == other.name
        return False
    
    def __str__(self):
        return self.name

    
# ---------------------------------------------------------------
# BAMDP Observation Model --> Fully observable state 
# ---------------------------------------------------------------
class BAMDPObservationModel(pomdp_py.ObservationModel):
    def probability(self, observation, next_state, action):
        if observation.name == next_state.name:
            return 1.0
        return 0.0
    
    def sample(self, next_state, action):
        return next_state.env_state

