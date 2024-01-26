
import pomdp_py


# we index the walls around a grid cell in
# clockwise fashion: top wall (0), right wall (1),
# bottom wall (2), left wall (3).
WALL = {
    0: "top",
    1: "right",
    2: "bottom",
    3: "left"
}

class Observation(pomdp_py.Observation):
    def __init__(self, walls, orientation):
        """
        Args:
            walls (tuple) is a tuple of integers, that indicate the walls
            around a grid cell that are present.
            orientation (float) is the orientation of the robot.
        """
        self.walls = walls
        self.orientation = orientation

    def __hash__(self):
        return hash((self.wall_case, self.orientation))

    def __eq__(self, other):
        if not isinstance(other, Observation):
            return False
        else:
            return self.walls == other.walls\
                and self.orientation == other.orientation
