import pomdp_py
import random
from pomdp_problems.tag.domain.action import *
from pomdp_problems.tag.models.transition_model import *

class TagPolicyModel(pomdp_py.RolloutPolicy):
    def __init__(self, grid_map=None):
        self._grid_map = grid_map

    def sample(self, state, **kwargs):
        return random.sample(self.get_all_actions(state, **kwargs), 1)[0]
    
    def get_all_actions(self, state=None, history=None):
        if state is not None:
            if self._grid_map is not None:
                valid_motions = self._grid_map.valid_motions(state.robot_position,
                                                             all_motions=MOTION_ACTIONS)
                return valid_motions | set({TagAction()})
        return MOTION_ACTIONS | set({TagAction()})

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

