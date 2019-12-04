from abc import ABC, abstractmethod

"""Because T, R, O may be different for the agent versus the environment,
it does not make much sense to have the POMDP class to hold this information;
instead, Agent should have its own T, R, O, pi and the Environment should
have its own T, R. The job of a POMDP is only to verify whether a given state,
action, or observation are valid."""

class POMDP(ABC):

    @abstractmethod
    def verify_state(self, state, **kwargs):
        pass

    @abstractmethod
    def verify_action(self, action, **kwargs):
        pass

    @abstractmethod
    def verify_observation(self, observation, **kwargs):
        pass    
