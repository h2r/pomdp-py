from pomdp_py.framework.basics cimport Action, Agent, Observation, State

cdef class Planner:

    cpdef public plan(self, Agent agent)
    cpdef public update(self, Action real_action, Observation real_observation)
    
cpdef sample_generative_model(Agent agent, State state, Action action)
