import pomdp_py
import random
import pomdp_problems.util as util
import pomdp_problems.tag.constants as constants
from pomdp_problems.tag.models.transition_model import TagTransitionModel


class TagTargetMotionPolicy(pomdp_py.GenerativeDistribution):
    def __init__(self,
                 grid_map,
                 pr_move_away=0.8,
                 pr_stay=0.2):
        self._grid_map = grid_map
        self._pr_move_away = pr_move_away
        self._pr_stay = pr_stay

    def _compute_candidate_actions(self,
                                   robot_position,
                                   target_position,
                                   valid_target_motion_actions):
        candidate_actions = set({})
        cur_dist = util.euclidean_dist(robot_position, target_position)
        for action in valid_target_motion_actions:
            next_target_position = TagTransitionModel.if_move_by(self._grid_map,
                                                                 target_position,
                                                                 action)
            next_dist = util.euclidean_dist(robot_position, next_target_position)
            if next_dist > cur_dist:
                candidate_actions.add(action)
        return candidate_actions
        

    def probability(self, next_target_position,
                    target_position, robot_position,
                    valid_target_motion_actions):
        # If it is impossible to go from target position to the next,
        # then it is a zero probability event.
        diff_x = abs(next_target_position[0] - target_position[0])
        diff_y = abs(next_target_position[1] - target_position[1])
        if not ((diff_x == 1 and diff_y == 0)
                or (diff_x == 0 and diff_y == 1)
                or (diff_x == 0 and diff_y == 0)):
            return constants.EPSILON

        if next_target_position == target_position:
            return self._pr_stay
        
        cur_dist = util.euclidean_dist(robot_position, target_position)
        next_dist = util.euclidean_dist(robot_position, next_target_position)
        candidate_actions = self._compute_candidate_actions(robot_position,
                                                            target_position,
                                                            valid_target_motion_actions)
        if len(candidate_actions) == 0:
            # No action can be taken, yet next_target_position is a valid
            # transition from current. That means the target is stuck. 
            return self._pr_move_away
        else:
            if next_dist <= cur_dist:
                # The target never actively moves closer to the robot
                return constants.EPSILON
            else:
                return self._pr_move_away / len(candidate_actions)

    def random(self, robot_position, target_position, valid_target_motion_actions,
               mpe=False):
        if mpe or random.uniform(0,1) > self._pr_stay:
            # Move away; Pick motion actions that makes the target moves away from the robot
            candidate_actions = self._compute_candidate_actions(robot_position,
                                                                target_position,
                                                                valid_target_motion_actions)
            if len(candidate_actions) == 0:
                return target_position
            
            chosen_action = random.sample(candidate_actions, 1)[0]
            return TagTransitionModel.if_move_by(self._grid_map,
                                                 target_position,
                                                 chosen_action)
        else:
            # stay
            return target_position

    def mpe(self, robot_position, target_position, valid_target_motion_actions):
        return self.random(robot_position, target_position,
                           valid_target_motion_actions,
                           mpe=True)
            
