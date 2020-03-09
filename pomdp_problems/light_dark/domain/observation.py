import pomdp_py

class Observation(pomdp_py.Observation):
    """Defines the Observation for the continuous light-dark domain;

    Observation space: 

        :math:`\Omega\subseteq\mathbb{R}^2` the observation of the robot is
            an estimate of the robot position :math:`g(x_t)\in\Omega`.

    """
    # the number of decimals to round up an observation when it is discrete.
    PRECISION=2
    
    def __init__(self, position, discrete=False):
        """
        Initializes a observation in light dark domain.

        Args:
            position (tuple): position of the robot.
        """
        self._discrete = discrete
        if len(position) != 2:
            raise ValueError("Observation position must be a vector of length 2")
        if self._discrete:
            self.position = position
        else:
            self.position = (round(position[0], Observation.PRECISION),
                             round(position[1], Observation.PRECISION))

    def discretize(self):
        return Observation(self.position, discrete=True)

    def __hash__(self):
        return hash(self.position)
    
    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.position == other.position
        else:
            return False
        
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return "Observation(%s)" % (str(self.position))
