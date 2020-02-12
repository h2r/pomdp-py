# Defines the State/Action/TransitionModel for the 2D Multi-Object Search domain;
#
# Origin: Multi-Object Search using Object-Oriented POMDPs (ICRA 2019)
# (extensions: action space changes,
#              different sensor model,
#              gridworld instead of topological graph)
#
# Description: Multi-Object Search in a 2D grid world.
#
# State space: S1 X S2 X ... Sn X Sr
#              where Si (1<=i<=n) is the object state, with attribute "pose" (x,y)
#              and Sr is the state of the robot, with attribute "pose" (x,y) and
#              "objects_found" (set).
#
# Action space: Motion U Look U Find
#               Motion Actions scheme 1: South, East, West, North.
#               Motion Actions scheme 2: Left 45deg, Right 45deg, Forward
#               motion actions move the robot deterministically.
#               Look: Interprets sensor input as observation
#               Find: Marks objects observed in the last Look action as
#                     (differs from original paper; reduces action space)
#               It is possible to force "Look" after every N/S/E/W action;
#               then the Look action could be dropped. This is optional behavior.
# 
# Transition: deterministic

import pomdp_py
import math
import copy

###### States ######
class TargetObjectState(pomdp_py.ObjectState):
    def __init__(self, objid, objclass, pose, res=1):
        super().__init__(objclass, {"pose":pose, "id":objid})
    def __str__(self):
        return '%s%s' % (str(self.objclass), str(self.pose))
    @property
    def pose(self):
        return self.attributes['pose']
    @property
    def objid(self):
        return self.attributes['id']

class RobotState(pomdp_py.ObjectState):
    def __init__(self, robot_id, pose, objects_found, camera_direction):
        """Note: camera_direction is None unless the robot is looking at a direction,
        in which case camera_direction is the string e.g. look+x, or 'look'"""
        super().__init__("robot", {"id":robot_id,
                                   "pose":pose,
                                   "objects_found": objects_found,
                                   "camera_direction": camera_direction})
    def __str__(self):
        return 'RobotState(%s%s|%s)' % (str(self.objclass), str(self.pose), str(self.objects_found))
    def __repr__(self):
        return str(self)
    @property
    def pose(self):
        return self.attributes['pose']
    @property
    def robot_pose(self):
        return self.attributes['pose']
    @property
    def objects_found(self):
        return self.attributes['objects_found']

class MosOOState(pomdp_py.OOState):
    def __init__(self, robot_id, object_states):
        self._robot_id = robot_id
        super().__init__(object_states)
    def get_robot_state(self):
        return self.object_states[self._robot_id]
    def object_pose(self, objid):
        return self.object_states[objid]["pose"]
    @property
    def robot_id(self):
        return self._robot_id
    @property
    def robot_pose(self):
        return self.object_states[self._robot_id]['pose']
    @property
    def object_poses(self):
        return {objid:self.object_states[objid]['pose']
                for objid in self.object_states
                if objid != self._robot_id}
    @property
    def robot_state(self):
        return self.object_states[self._robot_id]
    def __str__(self):
        return 'MosOOState(%d)%s' % (self._robot_id, str(self.object_states))
    def __repr__(self):
        return str(self)


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
    # scheme 1 (x,y)
    EAST = (1, 0)  # x is horizontal; x+ is right. y is vertical; y+ is up.
    WEST = (-1, 0)
    NORTH = (0, 1)
    SOUTH = (0, -1)
    # scheme 2 (vt, vw) translational, rotational velocities.
    FORWARD = (1, 0)
    BACKWARD = (-1, 0)
    LEFT = (0, -math.pi/4)  # left 45 deg
    RIGHT = (0, -math.pi/4) # right 45 deg

    def __init__(self, motion, scheme=MOTION_SCHEME):
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
    def __init__(self):
        super().__init__("look")
        
class FindAction(Action):
    def __init__(self):
        super().__init__("find")

####### Transition Model #######
class MosTransitionModel(pomdp_py.OOTransitionModel):
    """Object-oriented transition model"""
    def __init__(self, gridworld, epsilon=1e-9, for_env=False):
        """
        for_env (bool): True if this is a robot transition model used by the Environment.
             see RobotTransitionModel for details. 
        """
        self._gridworld = gridworld
        transition_models = {objid: StaticObjectTransitionModel(objid, epsilon=epsilon)
                             for objid in gridworld.target_objects}
        transition_models[gridworld.robot_id] = RobotTransitionModel(gridworld,
                                                                     epsilon=epsilon,
                                                                     for_env=for_env)
        super().__init__(transition_models)

    def sample(self, state, action, **kwargs):
        oostate = pomdp_py.OOTransitionModel.sample(self, state, action, **kwargs)
        return M3OOState(self._gridworld.robot_id, oostate.object_states)

    def argmax(self, state, action, normalized=False, **kwargs):
        oostate = pomdp_py.OOTransitionModel.argmax(self, state, action, **kwargs)
        return M3OOState(self._gridworld.robot_id, oostate.object_states)
    

class StaticObjectTransitionModel(pomdp_py.TransitionModel):
    """This model assumes the object is static."""
    def __init__(self, objid, epsilon=1e-9):
        self._objid = objid
        self._epsilon = epsilon

    def probability(self, next_object_state, state, action):
        if next_object_state != state.object_states[next_object_state['id']]:
            return self._epsilon
        else:
            return 1.0 - self._epsilon
    
    def sample(self, state, action):
        """Returns next_object_state"""
        return self.argmax(state, action)
    
    def argmax(self, state, action):
        """Returns the most likely next object_state"""
        return copy.deepcopy(state.object_states[self._objid])

    
class RobotTransitionModel(pomdp_py.TransitionModel):
    """We assume that the robot control is perfect and transitions are deterministic."""
    def __init__(self, gridworld, epsilon=1e-9, for_env=False):
        """
        for_env (bool): True if this is a robot transition model used by the Environment.
             The only difference is that the "detect" action will mark an object as
             detected if any voxel labeled by that object is within the viewing frustum.
             This differs from agent's RobotTransitionModel which will only mark an object
             as detected if the voxel at the object's pose is within the viewing frustum.
        """
        self._robot_id = gridworld.robot_id
        self._gridworld = gridworld
        self._epsilon = epsilon
        self._for_env = for_env

    def _expected_next_robot_pose(self, state, action):
        # IMPORTANT: If action is LookAction with motion, that means it is a look in a certain
        # direction, specified by `motion` from the default looking direction of -x. Therefore,
        # need to clear the angles of the robot; This is achieved by passing `absolute_rotation`
        # to if_move_by function.
        expected_robot_pose = self._gridworld.if_move_by(state.robot_pose, *action.motion,
                                                         object_poses=state.object_poses,
                                                         valid_pose_func=self._gridworld.valid_pose,
                                                         absolute_rotation=(isinstance(action, LookAction) and action.motion is not None))
        return expected_robot_pose

    def probability(self, next_robot_state, state, action):
        if next_robot_state != self.argmax(state, action):
            return self._epsilon
        else:
            return 1.0 - self._epsilon

    def argmax(self, state, action):
        """Returns the most likely next robot_state"""
        if isinstance(state, RobotState):
            robot_state = state
        else:
            robot_state = state.object_states[self._robot_id]
        # using shallow copy because we don't expect object state to reference other objects.            
        next_robot_state = copy.deepcopy(robot_state)
        next_robot_state['camera_direction'] = None  # camera direction is only not None when looking

        if isinstance(action, MotionAction):
            # motion action
            next_robot_state['pose'] = self._expected_next_robot_pose(state, action)

        elif isinstance(action, LookAction):
            if action.motion is not None:
                # rotate the robot
                next_robot_state['pose'] = self._expected_next_robot_pose(state, action)
            next_robot_state['camera_direction'] = action.name
                                                                      
        elif isinstance(action, DetectAction):
            # detect;
            object_poses = {objid:state.object_states[objid]['pose']
                            for objid in state.object_states
                            if objid != self._robot_id}
            # the detect action will mark all objects within the view frustum as detected.
            #   (the RGBD camera is always running and receiving point clouds)
            objects = self._gridworld.objects_within_view_range(robot_state['pose'],
                                                                object_poses, volumetric=self._for_env)
            next_robot_state['objects_found'] = tuple(set(next_robot_state['objects_found']) | set(objects))
        return next_robot_state
    
    def sample(self, state, action):
        """Returns next_robot_state"""
        return self.argmax(state, action)
