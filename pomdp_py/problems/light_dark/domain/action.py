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
        if len(control) != 2:
            raise ValueError("Action control must be a vector of length 2")        
        self.control = control

    def __hash__(self):
        return hash(self.control)
    
    def __eq__(self, other):
        if isinstance(other, Action):
            return self.control == other.control
        else:
            return False
        
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return "Action(%s)" % (str(self.control))
        
