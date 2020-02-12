# Reward model for Mos3D

import pomdp_py
from ..domain.action import *

class MosRewardModel(pomdp_py.RewardModel):
    def __init__(self, gridworld, big=1000, small=1):
        self.big = big
        self.small = small
        self._gridworld = gridworld
        
    def probability(self, reward, state, action, next_state, normalized=False, **kwargs):
        if reward == self._reward_func(state, action):
            return 1.0
        else:
            return 0.0
    def sample(self, state, action, next_state, normalized=False, **kwargs):
        # deterministic
        return self._reward_func(state, action, next_state)
    def argmax(self, state, action, next_state, normalized=False, **kwargs):
        """Returns the most likely reward"""
        return self._reward_func(state, action, next_state)

class GoalRewardModel(M3RewardModel):
    """
    This is a reward where the agent gets reward only for detect-related actions.
    """
    def __init__(self, gridworld, big=1000, small=1):
        super().__init__(gridworld, big=big, small=small, discount_factor=discount_factor)
        
    def _reward_func(self, state, action, next_state):
        reward = 0

        # If the robot has detected all objects
        if len(state.robot_state['objects_found']) == len(self._gridworld.target_objects):
            return 0  # no reward or penalty; the task is finished.
        
        if isinstance(action, MotionAction):
            reward = reward - self.small - action.distance_cost
        elif isinstance(action, LookAction):
            reward = reward - self.small
        elif isinstance(action, FindAction):

            if state.robot_state['camera_direction'] is None:
                # The robot didn't look before detect. So nothing is in the field of view.
                reward -= self.big
            else:
                # transition function should've taken care of the detection.
                new_objects_count = len(set(next_state.robot_state.objects_found) - set(state.robot_state.objects_found))
                if new_objects_count == 0:
                    # No new detection. "detect" is a bad action.
                    reward -= self.big
                else:
                    # Has new detection. Award.
                    reward += self.big
        return reward
    
