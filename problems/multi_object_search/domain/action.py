# Defines the Action for the 2D Multi-Object Search domain;
#
# Action space: Motion U Look U Find
#               Motion Actions scheme 1: South, East, West, North.
#               Motion Actions scheme 2: Left 45deg, Right 45deg, Forward
#               Look: Interprets sensor input as observation
#               Find: Marks objects observed in the last Look action as
#                     (differs from original paper; reduces action space)
#               It is possible to force "Look" after every N/S/E/W action;
#               then the Look action could be dropped. This is optional behavior.
import pomdp_py
import math

###### Actions ######
class Action(pomdp_py.Action):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other
    def __str__(self):
        return self.name
    def __repr__(self):
        return "Action(%s)" % self.name

MOTION_SCHEME="xy"  # can be either xy or vw
class MotionAction(Action):
    # scheme 1 (vx,vy,th)
    EAST = (1, 0, 0)  # x is horizontal; x+ is right. y is vertical; y+ is up.
    WEST = (-1, 0, math.pi)
    NORTH = (0, 1, math.pi/2)
    SOUTH = (0, -1, 3*math.pi/2)
    # scheme 2 (vt, vw) translational, rotational velocities.
    FORWARD = (1, 0)
    BACKWARD = (-1, 0)
    LEFT = (0, -math.pi/4)  # left 45 deg
    RIGHT = (0, math.pi/4) # right 45 deg

    def __init__(self, motion, scheme=MOTION_SCHEME, distance_cost=1):
        """
        motion (tuple): a tuple of floats that describes the motion;
        scheme (str): description of the motion scheme; Either
                      "xy" or "vw"
        """
        if scheme != "xy" and scheme != "vw":
            raise ValueError("Invalid motion scheme %s" % scheme)

        if scheme == "xy":
            if motion not in {MotionAction.EAST, MotionAction.WEST,
                              MotionAction.NORTH, MotionAction.SOUTH}:
                raise ValueError("Invalid move motion %s" % motion)
        else:
            if motion not in {MotionAction.FORWARD, MotionAction.BACKWARD,
                              MotionAction.LEFT, MotionAction.RIGHT}:
                raise ValueError("Invalid move motion %s" % motion)
            
        self.motion = motion
        self.scheme = scheme
        self.distance_cost = distance_cost
        super().__init__("move-%s-%s" % (scheme, str(motion)))
        
# Define some constant actions
MoveEast = MotionAction(MotionAction.EAST, scheme="xy")
MoveWest = MotionAction(MotionAction.WEST, scheme="xy")
MoveNorth = MotionAction(MotionAction.NORTH, scheme="xy")
MoveSouth = MotionAction(MotionAction.SOUTH, scheme="xy")
MoveForward = MotionAction(MotionAction.FORWARD, scheme="vw")
MoveBackward = MotionAction(MotionAction.BACKWARD, scheme="vw")
MoveLeft = MotionAction(MotionAction.LEFT, scheme="vw")
MoveRight = MotionAction(MotionAction.RIGHT, scheme="vw")

class LookAction(Action):
    # For simplicity, this LookAction is not parameterized by direction
    def __init__(self):
        super().__init__("look")
        
class FindAction(Action):
    def __init__(self):
        super().__init__("find")

if MOTION_SCHEME == "xy":
    ALL_ACTIONS = {MoveEast, MoveWest, MoveNorth, MoveSouth, LookAction, FindAction}
elif MOTION_SCHEME == "vw":
    ALL_ACTIONS = {MoveForward, MoveBackward, MoveLeft, MoveRight, LookAction, FindAction}    

