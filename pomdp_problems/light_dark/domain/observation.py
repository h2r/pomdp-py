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
        self.position = position
