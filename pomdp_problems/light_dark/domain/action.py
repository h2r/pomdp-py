"""Defines the Action for the continuous light-dark domain;

Origin: Belief space planning assuming maximum likelihood observations

Action space: 

    :math:`U\subseteq\mathbb{R}^2`. Quote from the paper: "The robot is
        modeled as a first-order system such that the robot velocity is determined
        by the control actions, :math:`u\in\mathbb{R}^2`.
"""
import pomdp_py

class Action(pomdp_py.Action):
    """The action is a vector of velocities"""
    def __init__(self, control):
        """
        Initializes a state in light dark domain.

        Args:
            control (tuple): velocity
        """
        self.control = control
