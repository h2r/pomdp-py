from abc import ABC, abstractmethod
from pomdp_py.framework.pomdp import POMDP

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

class ObjectState(ABC):
    def __init__(self, objclass, attributes):
        """
        class: "class",
        attributes: {
            "attr1": value,
            ...
        }
        """
        self.objclass = objclass
        self.attributes = attributes
        self._to_hash = pprint.pformat(self.attributes)   # automatically sorted by keys

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def __str__(self):
        return 'OOPOMDP_ObjectState::(%s,%s)' % (str(self.objclass),
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
    

class OOState(ABC):

    def __init__(self, object_states):
        """
        objects_states: dictionary of dictionaries; Each dictionary represents an object state:
            { ID: ObjectState }
        """
        # internally, objects are sorted by ID.
        self.object_states = object_states
        self._to_hash = pprint.pformat(self.object_states)  # automatically sorted by keys
        super().__init__(object_states)

    @abstractmethod        
    def __str__(self):
        return 'OOPOMDP_State::[%s]' % str(self.object_states)

    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        return self.object_states == other.object_states

    def __hash__(self):
        return hash(self._to_hash)
    
    @abstractmethod
    def __getitem__(self, index):
        raise NotImplemented
    
    @abstractmethod    
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
    
