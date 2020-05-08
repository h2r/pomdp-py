"""
The agent can take motion action and a look action.
"""

# Reuses the actions in the multi object search domain
import pomdp_py
from pomdp_problems.multi_object_search.domain.action\
    import MotionAction, MoveForward, MoveBackward, MoveLeft, MoveRight, LookAction
