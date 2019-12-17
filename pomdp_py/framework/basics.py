from abc import ABC, abstractmethod 

class Distribution(ABC):
    """
    A Distribution is a probability function that maps
    from variable value to a real value.
    """
    @abstractmethod
    def __getitem__(self, varval):
        raise NotImplemented
    @abstractmethod    
    def __setitem__(self, varval, value):
        raise NotImplemented
    def __hash__(self):
        return id(self)
    def __eq__(self, other):
        return id(self) == id(other)
    def __iter__(self):
        """Initialization of iterator over the values in this distribution"""
        raise NotImplemented
    def __next__(self):
        """Returns the next value of the iterator"""
        raise NotImplemented
    def probability(self, varval):
        return self[varval]

class GenerativeDistribution(Distribution):
    def mpe(self, **kwargs):
        raise NotImplemented
    def random(self, **kwargs):
        # Sample a state based on the underlying belief distribution
        raise NotImplemented
    def get_histogram(self):
        """Returns a dictionary from state to probability"""
        raise NotImplemented

class ObservationModel(ABC):
    @abstractmethod
    def probability(self, observation, next_state, action, **kwargs):
        raise NotImplemented
    @abstractmethod    
    def sample(self, next_state, action, **kwargs):
        """Returns observation"""
        raise NotImplemented
    @abstractmethod
    def argmax(self, next_state, action, **kwargs):
        """Returns the most likely observation"""
        raise NotImplemented
    def get_distribution(self, next_state, action, **kwargs):
        """Returns the underlying distribution of the model"""
        raise NotImplemented
    def get_all_observations(self):
        """Returns a set of all possible observations, if feasible."""
        raise NotImplemented        
    
class TransitionModel(ABC):
    @abstractmethod
    def probability(self, next_state, state, action, **kwargs):
        raise NotImplemented
    @abstractmethod    
    def sample(self, state, action, **kwargs):
        """Returns next_state"""
        raise NotImplemented
    @abstractmethod
    def argmax(self, state, action, **kwargs):
        """Returns the most likely next state"""
        raise NotImplemented
    def get_distribution(self, state, action, **kwargs):
        """Returns the underlying distribution of the model"""
        raise NotImplemented
    def get_all_states(self):
        """Returns a set of all possible states, if feasible."""
        raise NotImplemented    

class RewardModel(ABC):
    @abstractmethod
    def probability(self, reward, state, action, next_state, **kwargs):
        raise NotImplemented
    @abstractmethod    
    def sample(self, state, action, next_state, **kwargs):
        """Returns a reward"""
        raise NotImplemented
    @abstractmethod
    def argmax(self, state, action, next_state, **kwargs):
        """Returns the most likely reward"""
        raise NotImplemented
    def get_distribution(self, state, action, next_state, **kwargs):
        """Returns the underlying distribution of the model"""
        raise NotImplemented    

class BlackboxModel(ABC):
    @abstractmethod
    def sample(self, state, action, **kwargs):
        """Sample (s',o,r) ~ G(s',o,r)"""
        raise NotImplemented
    @abstractmethod
    def argmax(self, state, action, **kwargs):
        """Returns the most likely (s',o,r)"""
        raise NotImplemented

class PolicyModel(ABC):
    """The reason to have a policy model is to accommodate problems
    with very large action spaces, and the available actions may vary
    depending on the state (that is, certain actions have probabilty=0)"""
    @abstractmethod
    def probability(self, action, state, **kwargs):
        raise NotImplemented
    @abstractmethod    
    def sample(self, state, **kwargs):
        """Returns an action"""
        raise NotImplemented
    @abstractmethod
    def argmax(self, state, **kwargs):
        """Returns the most likely reward"""
        raise NotImplemented
    def get_distribution(self, state, **kwargs):
        """Returns the underlying distribution of the model"""
        raise NotImplemented
    def get_all_actions(self):
        """Returns a set of all possible actions, if feasible."""
        raise NotImplemented

# Belief distribution is just a distribution. There's nothing special,
# except that the update/abstraction function can be performed over them.
# But it would make the class hierarchy a lot more complicated if belief
# distribution is also made explicit, which means, for example, a belief
# distribution represented as a histogram would have to do multiple
# inheritance; doing so, the additional value is little.
