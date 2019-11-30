from abc import ABC, abstractmethod

class Planner(ABC):

    @abstractmethod
    def plan(self, agent):
        """The agent carries the information:
           Bt, ht, O,T,R/G, pi, necessary for planning"""
        raise NotImplemented
