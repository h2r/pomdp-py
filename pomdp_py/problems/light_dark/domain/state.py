"""Defines the State for the continuous light-dark domain;

Origin: Belief space planning assuming maximum likelihood observations

State space: 

    :math:`X\subseteq\mathbb{R}^2` the state of the robot
"""
import pomdp_py
import numpy as np

class State(pomdp_py.State):
    """The state of the problem is just the robot position"""
    def __init__(self, position):
        """
        Initializes a state in light dark domain.

        Args:
            position (tuple): position of the robot.
        """
        if len(position) != 2:
            raise ValueError("State position must be a vector of length 2")
        self.position = position

    def __hash__(self):
        return hash(self.position)
    
    def __eq__(self, other):
        if isinstance(other, State):
            return self.position == other.position
        else:
            return False
        
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return "State(%s)" % (str(self.position))
