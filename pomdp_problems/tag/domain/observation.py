"""The Tag problem. Implemented according to the paper `Anytime Point-Based
Approximations for Large POMDPs <https://arxiv.org/pdf/1110.0027.pdf>`_.

Observation space: the agent observes the target's location when the agent and
    the target are in the same cell.
"""
import pomdp_py

class TagObservation(pomdp_py.Observation):
    def __init__(self, target_position):
        self.target_position = target_position

    def __hash__(self):
        return hash(self.target_position)

    def __eq__(self, other):
        if not isinstance(other, TagObservation):
            return False
        else:
            return self.target_position == other.target_position

    def __str__(self):
        return 'Observation(%s)' % (str(self.target_position))

