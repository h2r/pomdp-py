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
from ..domain.state import *
from ..domain.action import *

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
