import pomdp_py
from pomdp_problems.tag.domain.state import *
from pomdp_problems.tag.models.transition_model import *
from pomdp_problems.tag.models.reward_model import *
from pomdp_problems.tag.models.components.motion_policy import *
from pomdp_problems.tag.models.components.grid_map import *
from pomdp_problems.multi_object_search.env.env import interpret
from pomdp_problems.multi_object_search.env.visual import MosViz

class TagEnvironment(pomdp_py.Environment):

    def __init__(self,
                 init_state,
                 grid_map,
                 pr_stay=0.2,
                 small=1,
                 big=10):
        self._grid_map = grid_map
        target_motion_policy = TagTargetMotionPolicy(grid_map,
                                                     pr_stay)
        transition_model = TagTransitionModel(grid_map, target_motion_policy)
        reward_model = TagRewardModel(small=small, big=big)
        super().__init__(init_state,
                         transition_model,
                         reward_model)

    @property
    def width(self):
        return self._grid_map.width
    
    @property
    def length(self):
        return self._grid_map.length

    @property
    def grid_map(self):
        return self._grid_map

    @classmethod
    def from_str(cls, worldstr, **kwargs):
        dim, robots, objects, obstacles, _ = interpret(worldstr)
        assert len(robots) == 1, "Does not support multiple robots."
        robot_position = robots[list(robots.keys())[0]].pose[:2]
        targets = []
        obstacle_poses = set({})
        for objid in objects:
            if objid not in obstacles:
                targets.append(objid)
            else:
                obstacle_poses.add(objects[objid].pose)
        assert len(targets) == 1, "Does not support multiple objects."                        
        target_position = objects[targets[0]].pose
        init_state = TagState(robot_position, target_position, False)
        grid_map = GridMap(dim[0], dim[1], obstacle_poses)
        return TagEnvironment(init_state, grid_map, **kwargs)
