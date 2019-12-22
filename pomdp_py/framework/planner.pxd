from pomdp_py.framework.basics cimport Action, Agent, Observation

cdef class Planner:

    cpdef public Action plan(self, Agent agent)
    cpdef public void update(self, Action real_action, Observation real_observation)
    
