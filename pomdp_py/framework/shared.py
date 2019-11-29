from abc import ABC, abstractmethod 

"""
The difference between a probability distribution and a 
probability model:
"""


class Distribution(ABC):
    """
    A Distribution is a probability function that maps
    from variable value to a real value.
    """
    @abstractmethod
    def __getitem__(self, var):
        pass
    @abstractmethod    
    def __setitem__(self, var, value):
        pass
    @abstractmethod    
    def __hash__(self):
        pass
    @abstractmethod    
    def __eq__(self, other):
        pass
    @abstractmethod    
    def __str__(self):
        pass

class GenerativeDistribution(Distribution):
    def mpe(self):
        pass
    def random(self):
        # Sample a state based on the underlying belief distribution
        pass
    def get_histogram(self):
        """Returns a dictionary from state to probability"""
        pass

class BeliefDistribution(GenerativeDistribution):
    def update(self, real_action, real_observation, pomdp, **kwargs):
        pass
    def get_abstraction(self, state_mapper):
        """Returns a representation of the distribution over abstracted states
        which can be used to initialize an instance of this kind of distribution"""
        pass

class ObservationModel(ABC):
    @abstractmethod
    def probability(self, observation, next_state, action, normalized=False):
        pass
    @abstractmethod    
    def sample(self, next_state, action, normalized=False):
        """Returns a tuple, (observation, probability) """
        pass
    @abstractmethod
    def argmax(self, next_state, action, normalized=False):
        """Returns the most likely observation"""
        pass
    @abstractmethod    
    def get_distribution(self, next_state, action):
        """Returns the underlying distribution of the observation model"""
        pass
    
class TransitionModel(ABC):
    @abstractmethod
    def probability(self, next_state, state, action, normalized=False):
        pass
    @abstractmethod    
    def sample(self, state, action, normalized=False):
        """Returns a tuple, (observation, probability) """
        pass
    @abstractmethod
    def argmax(self, state, action, normalized=False):
        """Returns the most likely observation"""
        pass
    @abstractmethod    
    def get_distribution(self, state, action):
        """Returns the underlying distribution of the observation model"""
        pass

class RewardModel(ABC):
    @abstractmethod
    def probability(self, reward, state, action, next_state, normalized=False):
        pass
    @abstractmethod    
    def sample(self, state, action, next_state, normalized=False):
        """Returns a tuple, (observation, probability) """
        pass
    @abstractmethod
    def argmax(self, state, action, next_state, normalized=False):
        """Returns the most likely observation"""
        pass
    @abstractmethod    
    def get_distribution(self, state, action, next_state):
        """Returns the underlying distribution of the observation model"""
        pass    
