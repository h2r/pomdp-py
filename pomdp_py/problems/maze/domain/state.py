"""Defines the State for the maze domain, which is the position of the robot and its orientation.
"""
import pomdp_py
import numpy as np

class State(pomdp_py.State):
    """The state of the problem is just the robot position"""
    def __init__(self, positition, orientation):
        """
        Initializes a state in light dark domain.

        Args:
            position (tuple): position of the robot.
        """
        if len(position) != 2:
            raise ValueError("State position must be a vector of length 2")
        self.position = positition
        self.orientation = orientation

    def __hash__(self):
        return hash(self.position, self.orientation)
    
    def __eq__(self, other):
        if isinstance(other, State):
            return self.position == other.position
        else:
            return False
        
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return "State(%s)" % (str(self.position, self.orientation))
