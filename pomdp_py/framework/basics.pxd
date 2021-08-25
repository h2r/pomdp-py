cdef class Distribution:
    pass
cdef class GenerativeDistribution(Distribution):
    pass
cdef class ObservationModel:
    pass
cdef class TransitionModel:
    pass
cdef class PolicyModel:
    pass
cdef class BlackboxModel:
    pass
cdef class RewardModel:
    pass

cdef class POMDP:
    cdef public Agent agent
    cdef public Environment env
    cdef public str name

cdef class Action:
    cdef public str name

cdef class State:
    pass

cdef class Observation:
    pass

cdef class Agent:
    cdef GenerativeDistribution _init_belief
    cdef PolicyModel _policy_model
    cdef TransitionModel _transition_model
    cdef RewardModel _reward_model
    cdef ObservationModel _observation_model
    cdef BlackboxModel _blackbox_model
    cdef GenerativeDistribution _cur_belief
    cdef tuple _history
    cdef dict __dict__

cdef class Environment:
    cdef State _init_state
    cdef TransitionModel _transition_model
    cdef RewardModel _reward_model
    cdef BlackboxModel _blackbox_model
    cdef State _cur_state

cdef class Option(Action):
    pass

cpdef sample_generative_model(Agent agent, State state, Action action, float discount_factor=*)
cpdef sample_explict_models(TransitionModel T, ObservationModel O, RewardModel R, State state, Action a, float discount_factor=*)
