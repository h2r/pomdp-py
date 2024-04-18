# cython: language_level=3, profile=True

from __future__ import annotations
from pomdp_py.framework.basics cimport (
    Agent,
    Environment,
    Observation,
    State,
    Action,
    TransitionModel,
    ObservationModel
)


cdef class Response:
    pass


cdef class ResponseModel:
    cdef Response _null_response


cdef class ResponseAgent(Agent):
    cdef ResponseModel _response_model


cdef class ResponseEnvironment(Environment):
    cdef ResponseModel _response_model


cpdef tuple[State, Observation, Response, int] sample_generative_model_with_response(
    TransitionModel T, ObservationModel O, ResponseModel R, State state, Action action,
    Response null_response, float discount_factor = *
)
