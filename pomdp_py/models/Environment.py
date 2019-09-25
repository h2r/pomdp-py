from abc import ABC, abstractmethod

class Environment(ABC):

    """An Environment maintains the true state of the world.
    For example, it is the 2D gridworld, rendered by pygame.
    Or it could be the 3D simulated world rendered by OpenGL.
    Therefore, when coding up an Environment, the developer
    should have in mind how to represent the state so that
    it can be used by a POMDP or OOPOMDP.
    """
    
    @property
    @abstractmethod
    def state(self):
        pass

    @abstractmethod
    def state_transition(self, action):
        """Modifies current state of the environment"""
        pass
