from pomdp_problems.tag.domain.action import *
from pomdp_problems.tag.models.transition_model import TagTransitionModel

class GridMap:

    def __init__(self, width, length, obstacle_poses):
        self.width = width
        self.length = length
        # set of obstacle poses
        self.obstacle_poses = obstacle_poses

    def valid_pose(self, position):
        if not (position[0] >= 0 and position[0] < width\
                and position[1] >= 0 and position[1] < length):
            return False
        if position in self.obstacle_poses:
            return False
        return True

    def valid_motions(self, position,
                      all_motions={MoveEast, MoveWest, MoveNorth, MoveSouth}):
        valid_motions = set({})
        for motion_action in all_motions:
            if TagTransitionModel.if_move_by(position, motion_action) == position:
                continue
            valid_motions.add(motion_action)
        return valid_motions

