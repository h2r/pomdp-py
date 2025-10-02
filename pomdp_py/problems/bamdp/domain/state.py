import pomdp_py 

class BAMDPState(pomdp_py.State):
    """State for BAMDP, which includes the location of the agent (discrete graph node) and the 
    model parameters (parameters of the Beta distribution --> counts of successes and failures).
    
    Attributes:
        agent_loc (str): the location of the agent in the graph.
        model_params (dict): a dictionary mapping actions to their corresponding model parameters.
            Each model parameter is a tuple of (success_count, failure_count).
    """

    def __init__(self, agent_loc, model_params):
        self.agent_loc = agent_loc
        self.model_params = model_params  # {action_name: (success_count, failure_count)}

    def __hash__(self):
        return hash((self.agent_loc, frozenset(self.model_params.items())))
    
    def __eq__(self, other):
        return (self.agent_loc == other.agent_loc and 
                self.model_params == other.model_params)