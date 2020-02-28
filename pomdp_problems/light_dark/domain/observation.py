"""Defines the Observation for the continuous light-dark domain;

Origin: Belief space planning assuming maximum likelihood observations

Observation space: 

    :math:`\Omega\subseteq\mathbb{R}^2` the observation of the robot is
        an estimate of the robot position :math:`g(x_t)\in\Omega`.

"""
import pomdp_py

class Observation(pomdp_py.Observation):
    """The observation of the problem is just the robot position"""
    def __init__(self, position):
        """
        Initializes a observation in light dark domain.

        Args:
            position (tuple): position of the robot.
        """
        if len(position) != 2:
            raise ValueError("Observation position must be a vector of length 2")
        self.position = position

    def __hash__(self):
        return hash(self.position)
    
    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.position == other.position
        else:
            return False
        
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return "Observation(%s)" % (str(self.position))
        
