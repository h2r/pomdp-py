from pomdp_problems.tag.domain.action import *
from pomdp_problems.tag.models.transition_model import TagTransitionModel
from pomdp_problems.multi_object_search.env.env import interpret

class GridMap:

    def __init__(self, width, length, obstacle_poses):
        self.width = width
        self.length = length
        # set of obstacle poses
        self.obstacle_poses = obstacle_poses

    def valid_pose(self, position):
        if not (position[0] >= 0 and position[0] < self.width\
                and position[1] >= 0 and position[1] < self.length):
            return False
        if position in self.obstacle_poses:
            return False
        return True

    def valid_motions(self, position,
                      all_motions=MOTION_ACTIONS):
        valid_motions = set({})
        for motion_action in all_motions:
            if TagTransitionModel.if_move_by(self, position, motion_action) == position:
                continue
            valid_motions.add(motion_action)
        return valid_motions

    @classmethod
    def from_str(cls, worldstr, **kwargs):
        dim, _, objects, obstacles, _ = interpret(worldstr)
        obstacle_poses = set({})
        for objid in objects:
            if objid in obstacles:
                obstacle_poses.add(objects[objid].pose)
        grid_map = GridMap(dim[0], dim[1], obstacle_poses)
        return grid_map

    def free_cells(self):
        cells = set({(x,y)
                     for x in range(self.width)
                     for y in range(self.length)
                     if (x,y) not in self.obstacle_poses})
        return cells
                
