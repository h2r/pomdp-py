"""Optional grid map to assist collision avoidance during planning."""

from pomdp_problems.multi_object_search.models.transition_model import RobotTransitionModel
from pomdp_problems.multi_object_search.domain.action import *
from pomdp_problems.multi_object_search.domain.state import *

class GridMap:
    """This map assists the agent to avoid planning invalid
    actions that will run into obstacles. Used if we assume
    the agent has a map. This map does not contain information
    about the object locations."""
    def __init__(self, width, length, obstacles):
        """
        Args:
            obstacles (dict): Map from objid to (x,y); The object is
                                   supposed to be an obstacle.
            width (int): width of the grid map
            length (int): length of the grid map 
        """
        self.width = width
        self.length = length
        self._obstacles = obstacles
        # An MosOOState that only contains poses for obstacles;
        # This is to allow calling RobotTransitionModel.if_move_by
        # function.
        self._obstacle_states = {
            objid: ObjectState(objid, "obstacle", self._obstacles[objid])
            for objid in self._obstacles
        }
        # set of obstacle poses
        self.obstacle_poses = set({self._obstacles[objid]
                                   for objid in self._obstacles})        

    def valid_motions(self, robot_id, robot_pose, all_motion_actions):
        """
        Returns a set of MotionAction(s) that are valid to
        be executed from robot pose (i.e. they will not bump
        into obstacles). The validity is determined under
        the assumption that the robot dynamics is deterministic.
        """
        state = MosOOState(self._obstacle_states)
        state.set_object_state(robot_id,
                               RobotState(robot_id, robot_pose, None, None))

        valid = set({})
        for motion_action in all_motion_actions:
            if not isinstance(motion_action, MotionAction):
                raise ValueError("This (%s) is not a motion action" % str(motion_action))

            next_pose = RobotTransitionModel.if_move_by(robot_id, state,
                                                        motion_action, (self.width, self.length))
            if next_pose != robot_pose:
                # robot moved --> valid motion
                valid.add(motion_action)
        return valid
