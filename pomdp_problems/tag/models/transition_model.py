"""The Tag problem. Implemented according to the paper `Anytime Point-Based
Approximations for Large POMDPs <https://arxiv.org/pdf/1110.0027.pdf>`_.

Transition model: the robot moves deterministically. The target's movement
    depends on the robot; With Pr=0.8 the target moves away from the robot,
    and with Pr=0.2, the target stays at the same place. The target never
    moves closer to the robot.
"""
import copy
import pomdp_py
import pomdp_problems.tag.constants as constants
from pomdp_problems.tag.domain.action import *

class TagTransitionModel(pomdp_py.TransitionModel):

    def __init__(self,
                 grid_map,
                 target_motion_policy):
        self._grid_map = grid_map
        self.target_motion_policy = target_motion_policy

    @classmethod
    def if_move_by(cls, grid_map, position, action):
        if isinstance(action, MotionAction):
            dx, dy = action.motion
            next_position = (position[0] + dx,
                             position[1] + dy)
            if grid_map.valid_pose(next_position):
                return next_position
        return position

    def probability(self, next_state, state, action, **kwargs):
        # Robot motion
        expected_robot_position = TagTransitionModel.if_move_by(self._grid_map,
                                                                state.robot_position,
                                                                action)
        if expected_robot_position != next_state.robot_position:
            return constants.EPSILON

        if isinstance(action, TagAction):
            if next_state.target_position == next_state.robot_position:
                if next_state.target_found:
                    return 1.0 - constants.EPSILON
                else:
                    return constants.EPSILON
            else:
                if next_state.target_found:
                    return constants.EPSILON
                else:
                    return 1.0 - constants.EPSILON

        # Target motion
        valid_target_motion_actions = self._grid_map.valid_motions(state.target_position)
        return self.target_motion_policy.probability(next_state.target_position,
                                                     state.target_position,
                                                     state.robot_position,
                                                     valid_target_motion_actions)

    def sample(self, state, action, argmax=False):
        # Robot motion
        next_state = copy.deepcopy(state)
        next_state.robot_position = TagTransitionModel.if_move_by(self._grid_map,
                                                                  state.robot_position,
                                                                  action)

        # If Tag action
        if isinstance(action, TagAction):
            if not state.target_found:
                if state.robot_position == state.target_position:
                    next_state.target_found = True
            return next_state

        # Target motion
        valid_target_motion_actions = self._grid_map.valid_motions(state.target_position)
        if not argmax:
            next_state.target_position = self.target_motion_policy.random(state.robot_position,
                                                                          state.target_position,
                                                                          valid_target_motion_actions)
        else:
            next_state.target_position = self.target_motion_policy.mpe(state.robot_position,
                                                                       state.target_position,
                                                                       valid_target_motion_actions)
        return next_state

    def argmax(self, state, action, **kwargs):
        return self.sample(state, action, argmax=True)
