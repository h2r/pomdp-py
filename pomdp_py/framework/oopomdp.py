from abc import abstractmethod
from pomdp_py.framework.pomdp import POMDP
from pomdp_py.framework.basics import TransitionModel, ObservationModel, GenerativeDistribution
import collections

"""
The addition of OOPOMDP versus POMDP is that states must be
instance of OOPOMDP_State which contains a map of object ID
to the corresponding object state, which must be an instance
of OOPOMDP_ObjectState. This makes the "verify_state" function
explicitly defined for OOPOMDP, given attributes and domains.
"""

class OOPOMDP(POMDP):

    def __init__(self, attributes, domains):
        self._attributes = attributes
        self._domains = domains

    def verify_state(self, state, **kwargs):
        if not isinstance(state, OOState):
            return False
        return self._verify_oostate(self._attributes, self._domains, state)

    def verify_object_state(self, object_state, **kwargs):
        objclass = object_state.objclass
        if objclass not in self._attributes:
            if verbose:
                print("Object class %s does not have specified attriubtes!" % objclass)
            return False
        attrs = object_state.attributes
        for attr in attrs:
            if attr not in self._attributes[objclass]:
                if verbose:
                    print("Attribute %s is not specified for class %s!" % (attr, objclass))
                return False
            attr_value = object_states[attr]
            if not domains[(objclass, attr)](attr_value):
                if verbose:
                    print("Attribute value %s not in domain (%s, %s)" % (attr_value, objclass, attr))
                return False
        return True

    def _verify_oostate(self, attributes, domains, state, verbose=True):
        """Returns true if state (OOPOMDP_State) is valid."""
        assert isinstance(state, OOPOMDP_State)
        object_states = state.object_states
        for objid in object_states:
            if not self.verify_object_state(object_states[objid]):
                return False
        return True    

    @abstractmethod
    def verify_action(self, action, **kwargs):
        pass

    @abstractmethod
    def verify_observation(self, observation, **kwargs):
        pass

    @abstractmethod    
    def verify_factored_observation(self, observation, **kwargs):
        """Since in OOPOMDP, observation function is factored by objects,
        or other entities (e.g. pixel or voxel), there could be observation
        per such entity."""
        pass    

class ObjectState:
    def __init__(self, objclass, attributes):
        """
        class: "class",
        attributes: {
            "attr1": value,  # value should be hashable
            ...
        }
        """
        self.objclass = objclass
        self.attributes = attributes
        self._to_hash = tuple((attr, hash(self.attributes[attr]))
                              for attr in self.attributes
                              if isinstance(self.attributes[attr], collections.Hashable))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '%s::(%s,%s)' % (str(self.__class__.__name__),
                                str(self.objclass),
                                str(self.attributes))
    
    def __hash__(self):
        return hash(self._to_hash)

    def __eq__(self, other):
        return self.objclass == other.objclass\
            and self.attributes == other.attributes

    def __getitem__(self, attr):
        return self.attributes[attr]

    def __setitem__(self, attr, value):
        self.attributes[attr] = value
    
    def __len__(self):
        return len(self.attributes)

    def copy(self):
        return OOPOMDP_ObjectState(self.objclass, copy.deepcopy(self.attributes))
    

class OOState:

    def __init__(self, object_states):
        """
        objects_states: dictionary of dictionaries; Each dictionary represents an object state:
            { ID: ObjectState }
        """
        # internally, objects are sorted by ID.
        self.object_states = object_states
        self._to_hash = tuple((objid, hash(self.object_states[objid]))
                              for objid in self.object_states)
        # self._to_hash = pprint.pformat(self.object_states)  # automatically sorted by keys

    def __str__(self):
        return '%s::[%s]' % (str(self.__class__.__name__),
                             str(self.object_states))

    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        return self.object_states == other.object_states

    def __hash__(self):
        return hash(self._to_hash)
    
    def __getitem__(self, index):
        raise NotImplemented

    def __len__(self):
        raise NotImplemented

    def get_object_state(self, objid):
        return self.object_states[objid]

    def set_object_state(self, objid, object_state):
        self.object_states[objid] = object_state

    def get_object_class(self, objid):
        return self.object_states[objid].objclass

    def get_object_attribute(self, objid, attr):
        return self.object_states[objid][attr]

    def copy(self):
        return OOPOMDP_State(copy.deepcopy(self.object_states))

class OOTransitionModel(TransitionModel):

    """
    T(s' | s, a) = Pi T(si' | s, a)
    """

    def __init__(self, transition_models):
        """
        transition_models (dict) objid -> transition_model
        """
        self._transition_models = transition_models
    
    def probability(self, next_state, state, action, **kwargs):
        if not isinstance(next_state, OOState):
            raise ValueError("next_state must be OOState")
        if not isinstance(state, OOState):
            raise ValueError("state must be OOState")
        if next_state.object_states.keys() != state.object_states.keys():
            raise ValueError("object types modified between states")
        trans_prob = 1.0
        for objid in state.object_states:
            object_state = state.object_states[objid]
            next_object_state = next_state.object_states[objid]
            trans_prob = trans_prob * self._transition_models[objid].probability(next_object_state, state, action, **kwargs)
        return trans_prob

    def sample(self, state, action, **kwargs):
        """Returns next_state"""
        if not isinstance(state, OOState):
            raise ValueError("state must be OOState")
        object_states = {}
        for objid in state.object_states:
            next_object_state = self._transition_models[objid].sample(state, action, **kwargs)
            object_states[objid] = next_object_state
        return OOState(object_states)

    def argmax(self, state, action, normalized=False, **kwargs):
        """Returns the most likely next state"""
        if not isinstance(state, OOState):
            raise ValueError("state must be OOState")
        object_states = {}
        for objid in state.object_states:
            next_object_state = self._transition_models[objid].argmax(state, action, **kwargs)
            object_states[objid] = next_object_state
        return OOState(object_states)

    def __getitem__(self, objid):
        return self._transition_models[objid]    


class OOObservationModel(ObservationModel):

    """
    O(z | s', a) = Pi T(zi' | s', a)
    """    

    def __init__(self, observation_models, factor_observation_func, merge_observations_func):
        """
        observation_models (dict) objid -> observation_model
        factor_observation_func: (observation, objid, next_state, action) -> {objid->observations}
        merge_observations_func: (factored_observations, next_state, action) -> observation
        """
        self._observation_models = observation_models
        self._factor_observation_func = factor_observation_func
        self._merge_observations_func = merge_observations_func
    
    def probability(self, observation, next_state, action, **kwargs):
        if not isinstance(next_state, OOState):
            raise ValueError("state must be OOState")
        obsrv_prob = 1.0
        factored_observations = self._factor_observation_func(observation, next_state)
        for objid in next_state.object_states:
            obsrv_prob = obsrv_prob * self._observation_models[objid].probability(factored_observations[objid],
                                                                                  next_state, action, **kwargs)
        return obsrv_prob

    def sample(self, next_state, action, argmax=False, **kwargs):
        """Returns observation"""
        if not isinstance(next_state, OOState):
            raise ValueError("state must be OOState")
        factored_observations = {}
        for objid in next_state.object_states:
            if not argmax:
                observation = self._observation_models[objid].sample(next_state,
                                                                     action, **kwargs)
            else:
                observation = self._observation_models[objid].argmax(next_state,
                                                                     action, **kwargs)                
            factored_observations[objid] = observation
        return self._merge_observations_func(factored_observations, next_state, **kwargs)

    def argmax(self, next_state, action, **kwargs):
        """Returns observation"""
        return self.sample(next_state, action, argmax=True, **kwargs)

    def __getitem__(self, objid):
        return self._observation_models[objid]


class OOBelief(GenerativeDistribution):
    def __init__(self, object_beliefs):
        """
        object_beliefs (objid -> GenerativeDistribution)
        """
        self._object_beliefs = object_beliefs

    def __getitem__(self, state):
        if not isinstance(state, OOState):
            raise ValueError("state must be OOState")
        belief_prob = 1.0
        for objid in state.object_states:
            object_state = state.object_states[objid]
            belief_prob = belief_prob * self._object_beliefs[objid].probability(object_state)
        return belief_prob

    def mpe(self, **kwargs):
        object_states = {}
        for objid in self._object_beliefs:
            object_states[objid] = self._object_beliefs[objid].mpe(**kwargs)
        return OOState(object_states)
    
    def random(self, **kwargs):
        object_states = {}
        for objid in self._object_beliefs:
            object_states[objid] = self._object_beliefs[objid].random(**kwargs)
        return OOState(object_states)        
    
    def __setitem__(self, oostate, value):
        raise NotImplemented
        
    def object_belief(self, objid):
        return self._object_beliefs[objid]

    def set_object_belief(self, objid, belief):
        self._object_beliefs[objid] = belief
