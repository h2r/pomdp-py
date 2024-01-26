"""
Defines the Action for the 2D Multi-Object Search domain;

Action space: 

    Motion :math:`\cup` Look :math:`\cup` Find

* Motion Actions scheme 1: South, East, West, North.
* Motion Actions scheme 2: Left 45deg, Right 45deg, Forward
* Look: Interprets sensor input as observation
* Find: Marks objects observed in the last Look action as
  (differs from original paper; reduces action space)

It is possible to force "Look" after every N/S/E/W action;
then the Look action could be dropped. This is optional behavior.
"""
import pomdp_py
import math

###### Actions ######
class Action(pomdp_py.Action):
    """Mos action; Simple named action."""
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
STEP_SIZE=1
class MotionAction(Action):
    # scheme 1 (vx,vy,th)
    SCHEME_XYTH = "xyth"
    EAST = (STEP_SIZE, 0, 0)  # x is horizontal; x+ is right. y is vertical; y+ is down.
    WEST = (-STEP_SIZE, 0, math.pi)
    NORTH = (0, -STEP_SIZE, 3*math.pi/2)
    SOUTH = (0, STEP_SIZE, math.pi/2)
    
    # scheme 2 (vt, vw) translational, rotational velocities.
    SCHEME_VW = "vw" 
    FORWARD = (STEP_SIZE, 0)
    BACKWARD = (-STEP_SIZE, 0)
    LEFT = (0, -math.pi/4)  # left 45 deg
    RIGHT = (0, math.pi/4) # right 45 deg

    # scheme 3 (vx,vy)
    SCHEME_XY = "xy"
    EAST2D = (STEP_SIZE, 0)  # x is horizontal; x+ is right. y is vertical; y+ is down.
    WEST2D = (-STEP_SIZE, 0)
    NORTH2D = (0, -STEP_SIZE)
    SOUTH2D = (0, STEP_SIZE)

    SCHEMES = {"xyth", "xy", "vw"}

    def __init__(self, motion,
                 scheme=MOTION_SCHEME, distance_cost=1,
                 motion_name=None):
        """
        motion (tuple): a tuple of floats that describes the motion;
        scheme (str): description of the motion scheme; Either
                      "xy" or "vw"
        """
        if scheme not in MotionAction.SCHEMES:
            raise ValueError("Invalid motion scheme %s" % scheme)

        if scheme == MotionAction.SCHEME_XYTH:
            if motion not in {MotionAction.EAST, MotionAction.WEST,
                              MotionAction.NORTH, MotionAction.SOUTH}:
                raise ValueError("Invalid move motion %s" % str(motion))
        elif scheme == MotionAction.SCHEME_VW:
            if motion not in {MotionAction.FORWARD, MotionAction.BACKWARD,
                              MotionAction.LEFT, MotionAction.RIGHT}:
                raise ValueError("Invalid move motion %s" % str(motion))
        elif scheme == MotionAction.SCHEME_XY:
            if motion not in {MotionAction.EAST2D, MotionAction.WEST2D,
                              MotionAction.NORTH2D, MotionAction.SOUTH2D}:
                raise ValueError("Invalid move motion %s" % str(motion))
            
        self.motion = motion
        self.scheme = scheme
        self.distance_cost = distance_cost
        if motion_name is None:
            motion_name = str(motion)
        super().__init__("move-%s-%s" % (scheme, motion_name))
        
# Define some constant actions
MoveEast = MotionAction(MotionAction.EAST,   scheme=MotionAction.SCHEME_XYTH, motion_name="East")
MoveWest = MotionAction(MotionAction.WEST,   scheme=MotionAction.SCHEME_XYTH, motion_name="West")
MoveNorth = MotionAction(MotionAction.NORTH, scheme=MotionAction.SCHEME_XYTH, motion_name="North")
MoveSouth = MotionAction(MotionAction.SOUTH, scheme=MotionAction.SCHEME_XYTH, motion_name="South")

MoveForward  = MotionAction(MotionAction.FORWARD,  scheme=MotionAction.SCHEME_VW, motion_name="Forward")
MoveBackward = MotionAction(MotionAction.BACKWARD, scheme=MotionAction.SCHEME_VW, motion_name="Backward")
MoveLeft     = MotionAction(MotionAction.LEFT,     scheme=MotionAction.SCHEME_VW, motion_name="TurnLeft")
MoveRight    = MotionAction(MotionAction.RIGHT,    scheme=MotionAction.SCHEME_VW, motion_name="TurnRight")

MoveEast2D  = MotionAction(MotionAction.EAST2D,  scheme=MotionAction.SCHEME_XY, motion_name="East2D")
MoveWest2D  = MotionAction(MotionAction.WEST2D,  scheme=MotionAction.SCHEME_XY, motion_name="West2D")
MoveNorth2D = MotionAction(MotionAction.NORTH2D, scheme=MotionAction.SCHEME_XY, motion_name="North2D")
MoveSouth2D = MotionAction(MotionAction.SOUTH2D, scheme=MotionAction.SCHEME_XY, motion_name="South2D")


class LookAction(Action):
    # For simplicity, this LookAction is not parameterized by direction
    def __init__(self):
        super().__init__("look")
        
class FindAction(Action):
    def __init__(self):
        super().__init__("find")

Look = LookAction()
Find = FindAction()

if MOTION_SCHEME == "xy":
    ALL_MOTION_ACTIONS = {MoveEast, MoveWest, MoveNorth, MoveSouth}
elif MOTION_SCHEME == "vw":
    ALL_MOTION_ACTIONS = {MoveForward, MoveBackward, MoveLeft, MoveRight}
else:
    raise ValueError("motion scheme '%s' is invalid" % MOTION_SCHEME)
