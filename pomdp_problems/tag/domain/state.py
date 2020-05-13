"""The Tag problem. Implemented according to the paper `Anytime Point-Based
Approximations for Large POMDPs <https://arxiv.org/pdf/1110.0027.pdf>`_.

State space: state of the robot (x,y), state of the person (x,y), person found.

"""
import pomdp_py

class TagState(pomdp_py.State):
    
    def __init__(self, robot_position, target_position, target_found):
        """
        robot_position (tuple): x,y location of the robot.
        target_position (tuple): x,y location of the target.
        target_found (bool): True if the target is found.
        """
        self.robot_position = robot_position
        self.target_position = target_position
        self.target_found = target_found

    def __hash__(self):
        return hash((self.robot_position, self.target_position, self.target_found))

    def __eq__(self, other):
        if not isinstance(other, TagState):
            return False
        else:
            return self.robot_position == other.robot_position\
                and self.target_position == other.target_position\
                and self.target_found == other.target_found

    def __str__(self):
        return 'State(%s, %s | %s)' % (str(self.robot_position),
                                       str(self.target_position),
                                       str(self.target_found))

    def __repr__(self):
        return str(self)
