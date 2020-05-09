import pomdp_py
from pomdp_problems.tag.domain.observation import *
import pomdp_problems.tag.constants as constants

class TagObservationModel(pomdp_py.ObservationModel):
    """In this observation model, the robot deterministically
    observes the target location when it is in the same grid cell
    as the target. Ohterwise the robot does not observe anything."""

    def probability(self, observation, next_state, action, **kwargs):
        if next_state.robot_position == next_state.target_position:
            if observation.target_position is None:
                return constants.EPSILON
            else:
                if observation.target_position == next_state.target_position:
                    return 1.0 - constants.EPSILON
                else:
                    return constants.EPSILON
        else:
            if observation.target_position is None:
                return 1.0 - constants.EPSILON
            else:
                return constants.EPSILON
    
    def sample(self, next_state, action):
        """There is no stochaisticity in the observation model"""
        if next_state.robot_position == next_state.target_position:
            return TagObservation(next_state.target_position)
        else:
            return TagObservation(None)

    def argmax(self, next_state, action):
        return self.sample(next_state, action)
