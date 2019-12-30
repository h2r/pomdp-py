from pomdp_py.framework.basics cimport Action, Agent, Observation, State

cdef class Planner:

    cpdef public plan(self, Agent agent):    
        """The agent carries the information:
        Bt, ht, O,T,R/G, pi, necessary for planning"""
        raise NotImplementedError

    cpdef public update(self, Action real_action, Observation real_observation):
        """Updates the planner based on real action and observation.
        Updates the agent accordingly if necessary. If the agent's
        belief is also updated here, the `update_agent_belief`
        attribute should be set to True. By default, does nothing."""
        pass    

    def updates_agent_belief(self):
        """True if planner's update function also updates agent's
        belief."""
        return False

cpdef sample_generative_model(Agent agent, State state, Action action):
    '''
    (s', o, r) ~ G(s, a)
    '''
    cdef State next_state
    cdef Observation observation
    cdef float reward        

    if agent.transition_model is None:
        next_state, observation, reward = agent.generative_model.sample(state, action)
    else:
        next_state = agent.transition_model.sample(state, action)
        observation = agent.observation_model.sample(next_state, action)
        reward = agent.reward_model.sample(state, action, next_state)
    return next_state, observation, reward
