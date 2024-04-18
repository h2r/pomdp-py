# cython: language_level=3, profile=True

from __future__ import annotations
from pomdp_py.framework.basics cimport (
    Agent,
    GenerativeDistribution,
    PolicyModel,
    TransitionModel,
    ObservationModel,
    BlackboxModel,
    Action,
    Observation,
    State,
    Environment,
    Option
)
from typing import Optional


cdef class Response:
    """
    A Response class maintains variables within a dictionary.
    However, subclasses of Response can provide access to the
    dictionary variables using the dot (.) operator. Currently,
    this class can handle arithmetic and comparison operations.
    However, if special operations will need to be performed,
    these operations need to be handled in the subclass.
    """

    def copy(self) -> Response:
        raise NotImplementedError

    @staticmethod
    def null() -> Response:
        raise NotImplementedError

    def __add__(self, other: Response) -> Response:
        raise NotImplementedError

    def __radd__(self, other: Response) -> Response:
        return self.__add__(other)

    def __mul__(self, other: float | int) -> Response:
        raise NotImplementedError

    def __rmul__(self, other: float | int) -> Response:
        return self.__mul__(other)

    def __eq__(self, other: Response) -> bool:
        raise NotImplementedError

    def __ne__(self, other: Response) -> bool:
        raise NotImplementedError

    def __lt__(self, other: Response) -> bool:
        raise NotImplementedError

    def __le__(self, other: Response) -> bool:
        raise NotImplementedError

    def __gt__(self, other: Response) -> bool:
        raise NotImplementedError

    def __ge__(self, other: Response) -> bool:
        raise NotImplementedError
        
    def __str__(self) -> str:
        raise NotImplementedError


cdef class ResponseModel:
    """
    A ResponseModel returns a real or simulated response after the agent interacts with 
    the real or a simulated environment. The implementation of this model contains a 
    collection of more specific models such as reward and cost models.
    """
    def __init__(self):
        pass

    def null_response(self) -> Response:
        raise NotImplementedError

    def sample(self, state: State, action: Action, next_state: State) -> Response:
        raise NotImplementedError


cdef class ResponseAgent(Agent):
    """
    A `ResponseAgent` behaves the same as an `Agent` with one difference: a
    `ReponseAgent` adds a `ResponseModel`. The `ResponseAgent` also provides direct
    access to the models maintained in the `ResponseModel` to reduce the wordiness of
    the code.
    """

    def __init__(
        self,
        init_belief: GenerativeDistribution,
        policy_model: Optional[PolicyModel] = None,
        transition_model: Optional[TransitionModel] = None,
        observation_model: Optional[ObservationModel] = None,
        response_model: Optional[ResponseModel] = None,
        blackbox_model: Optional[BlackboxModel] = None,
        name: Optional[str] = None
    ):
        super().__init__(
            init_belief=init_belief,
            policy_model=policy_model,
            transition_model=transition_model,
            observation_model=observation_model,
            reward_model=None,
            blackbox_model=blackbox_model,
        )

        if (
            not isinstance(response_model, ResponseModel)
            and response_model is not None
        ):
            raise TypeError(
                "response_model must be type ResponseModel, "
                f"but got type {type(response_model)}."
            )
        self._response_model = None
        if response_model is not None:
            self.set_response_model(response_model)

    @property
    def reward_model(self):
        raise AttributeError(
            "Use the response_model property to access the reward model."
        )

    @property
    def response_model(self) -> ResponseModel:
        if self._response_model is None:
            raise ValueError(
                "response_model is None. Call set_response_model to set a model."
            )
        return self._response_model

    def set_response_model(self, response_model: ResponseModel) -> None:
        if not isinstance(response_model, ResponseModel):
            raise TypeError(
                f"model must be type ResponseModel, but got type {type(response_model)}."
            )
        self._response_model = response_model


cdef class ResponseEnvironment(Environment):

    def __init__(
        self,
        init_state: State,
        transition_model: Optional[TransitionModel] = None,
        response_model: Optional[ResponseModel] = None,
        blackbox_model: Optional[BlackboxModel] = None
    ) -> None:
        super().__init__(
            init_state=init_state,
            transition_model=transition_model,
            reward_model=None,
            blackbox_model=blackbox_model,
        )
        if response_model is not None and blackbox_model is not None:
            raise ValueError(
                "Cannot specify a response and blackbox model at the same time."
            )
        self._response_model = response_model

    @property
    def reward_model(self):
        raise AttributeError(
            "Use the response_model property to access the reward model."
        )

    @property
    def response_model(self) -> ResponseModel:
        """
        Returns:
            The ResponseModel.
        """
        return self._response_model

    def set_models(
        self,
        transition_model: Optional[TransitionModel] = None,
        response_model: Optional[ResponseModel] = None,
        blackbox_model: Optional[BlackboxModel] = None,
    ) -> None:
        """
        Reassigns the models to be the ones given.

        Args:
            transition_model (TransitionModel): The transition model.
            response_model (ResponseModel): The response model.
            blackbox_model (BlackboxModel): Provided when the transition model and
                response model are not available.

        Returns:
            None
        """
        super().set_models(
            transition_model=transition_model,
            reward_model=None,
            blackbox_model=blackbox_model,
        )
        if response_model is not None and blackbox_model is not None:
            raise ValueError(
                "Cannot specify a response and blackbox model at the same time."
            )
        self._response_model = response_model

    def state_transition(
        self,
        action: Action,
        execute: bool = True,
        discount_factor: float = 1.0
    ) -> Response | tuple[State, Response]:
        """
        Simulates a state transition given `action`. If `execute` is set to True,
        then the resulting state will be the new current state of the environment.

        Args:
            action (Action): action that triggers the state transition.
            execute (bool): If True, the resulting state of the transition will become
                            the current state.
            discount_factor (float): Only necessary if action is an Option. It is the
                discount factor when executing actions following an option's policy
                until reaching terminal condition.

        Returns:
            Response or tuple[State, Response]: reward as a result of `action` and state
            transition, if `execute` is True (next_state, reward) if `execute` is False.
        """
        next_state, response, _ = sample_generative_model_with_response(
            T=self.transition_model,
            O=None,
            R=self.response_model,
            state=self.state,
            action=action,
            null_response=self.response_model.null_response(),
            discount_factor=discount_factor
        )

        if execute:
            self.apply_transition(next_state)
            return response
        else:
            return next_state, response


cpdef tuple[State, Observation, Response, int] sample_generative_model_with_response(
    TransitionModel T,
    ObservationModel O,
    ResponseModel R,
    State state,
    Action action,
    Response null_response,
    float discount_factor = 1.0
):
    cdef State next_state
    cdef Observation observation
    cdef Response response = null_response
    cdef Option option
    cdef int nsteps = 0

    if isinstance(action, Option):
        # The action is an option; simulate a rollout of the option
        option = action
        if not option.initiation(state):
            # state is not in the initiation set of the option. This is
            # similar to the case when you are in a particular (e.g. terminal)
            # state and certain action cannot be performed, which will still
            # be added to the PO-MCTS tree because some other state with the
            # same history will allow this action. In this case, that certain
            # action will lead to no state change, no observation, and 0 reward,
            # because nothing happened.
            if O is not None:
                return state, None, 0, 0
            else:
                return state, 0, 0

        step_discount_factor = 1.0
        while not option.termination(state):
            action = option.sample(state)
            next_state = T.sample(state, action)
            # For now, we don't care about intermediate observations (future work?).
            response += step_discount_factor * R.sample(state, action, next_state)
            step_discount_factor *= discount_factor
            state = next_state
            nsteps += 1
        # sample observation at the end, where action is the last action.
        # (doesn't quite make sense to just use option as the action at this point.)
    else:
        next_state = T.sample(state, action)
        response = R.sample(state, action, next_state)
        nsteps += 1
    if O is not None:
        observation = O.sample(next_state, action)
        return next_state, observation, response, nsteps
    else:
        return next_state, response, nsteps
