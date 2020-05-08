"""The Tag problem. Implemented according to the paper `Anytime Point-Based
Approximations for Large POMDPs <https://arxiv.org/pdf/1110.0027.pdf>`_.

Action space: The agent can take motion action and a tag action.
"""

# Reuses the actions in the multi object search domain
import pomdp_py
from pomdp_problems.multi_object_search.domain.action\
    import Action, MotionAction, MoveForward, MoveBackward, MoveLeft, MoveRight, LookAction

class TagAction(Action):
    def __init__(self):
        super().__init__("tag")    
