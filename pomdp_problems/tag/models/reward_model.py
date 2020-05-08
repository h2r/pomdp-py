import pomdp_py
from pomdp_problems.tag.domain.action import *

class TagRewardModel(pomdp_py.RewardModel):

    def __init__(self, small=1, big=10):
        self.small = small
        self.big = big

    def probability(self, reward, state, action, next_state, normalized=False, **kwargs):
        if reward == self._reward_func(state, action):
            return 1.0
        else:
            return 0.0
        
    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action, next_state)

    def _reward_func(self, state, action, next_state):
        if isinstance(action, MotionAction):
            return -self.small
        else:
            # Tag action
            assert isinstance(action, TagAction)
            if next_state.target_position == next_state.robot_position:
                if next_state.target_found:
                    return self.big
            return -self.big
