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

    def probability(self, next_target_position,
                    target_position, robot_position,
                    valid_target_motion_actions):
        if next_state.target_position == state.target_position:
            return self._pr_stay
        
        cur_dist = util.euclidean_dist(robot_position, target_position)
        next_dist = util.euclidean_dist(robot_position, next_target_position)
        if next_dist < cur_dist:
            return constants.EPSILON
        else:
            assert next_dist > cur_dist
            return self.pr_move_away / len(valid_motion_actions)

    def random(self, robot_position, target_position, valid_target_motion_actions,
               mpe=False):
        if mpe or random.uniform(0,1) > self._pr_stay:
            # Move away; Pick motion actions that makes the target moves away from the robot
            candidate_actions = set({})
            cur_dist = util.euclidean_dist(robot_position, target_position)
            for action in valid_target_motion_actions:
                next_target_position = TagTransitionModel.if_move_by(self._grid_map,
                                                                     target_position,
                                                                     action)
                next_dist = util.euclidean_dist(robot_position, next_target_position)
                if next_dist > cur_dist:
                    candidate_actions.add(action)
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
            
