"""Defines the State for the continuous light-dark domain;

Origin: Belief space planning assuming maximum likelihood observations

State space: 

    :math:`X\subseteq\mathbb{R}^2` the state of the robot

"""
import pomdp_py

class State(pomdp_py.State):
    """The state of the problem is just the robot position"""
    def __init__(self, position):
        """
        Initializes a state in light dark domain.

        Args:
            position (tuple): position of the robot.
        """
        self.position = position
