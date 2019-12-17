from abc import ABC, abstractmethod

class Planner(ABC):

    @abstractmethod
    def plan(self, agent, **kwargs):
        """The agent carries the information:
           Bt, ht, O,T,R/G, pi, necessary for planning"""
        raise NotImplemented

    def update(self, real_action, real_observation):
        """Updates the planner based on real action and observation.
        Updates the agent accordingly if necessary. If the agent's
        belief is also updated here, the `update_agent_belief`
        attribute should be set to True. By default, does nothing."""
        pass

    @property
    def updates_agent_belief(self):
        """True if planner's update function also updates agent's
        belief."""
        return False
