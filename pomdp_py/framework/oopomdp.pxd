from pomdp_py.framework.basics cimport POMDP, State, Action, Observation,\
    ObservationModel, TransitionModel, GenerativeDistribution

cdef class OOPOMDP(POMDP):
    pass

cdef class ObjectState(State):
    cdef public str objclass
    cdef public dict attributes
    cdef int _hashcache

cdef class OOState(State):
    cdef public dict object_states
    cdef frozenset _situation
    cdef int _hashcache

cdef class OOTransitionModel(TransitionModel):
    cdef dict _transition_models

cdef class OOObservation(Observation):
    pass

cdef class OOObservationModel(ObservationModel):
    cdef dict _observation_models

cdef class OOBelief(GenerativeDistribution):
    cdef dict _object_beliefs
