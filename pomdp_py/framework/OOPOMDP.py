from abc import ABC, abstractmethod

"""
The addition of OOPOMDP versus POMDP is that states must be
instance of OOPOMDP_State which contains a map of object ID
to the corresponding object state, which must be an instance
of OOPOMDP_ObjectState. This makes the "verify_state" function
explicitly defined for OOPOMDP, given attributes and domains.
"""

class OOPOMDP(ABC):

    def verify_state(cls, attributes, domains, state, verbose=True):
        """Returns true if state (OOPOMDP_State) is valid."""
        assert isinstance(state, OOPOMDP_State)
        object_states = state.object_states
        for objid in object_states:
            objclass = object_states[objid].objclass
            if objclass not in attributes:
                if verbose:
                    print("Object class %s does not have specified attriubtes!" % objclass)
                return False
            attrs = object_states[objid].attributes
            for attr in attrs:
                if attr not in attributes[objclass]:
                    if verbose:
                        print("Attribute %s is not specified for class %s!" % (attr, objclass))
                    return False
                attr_value = object_states[objid][attr]
                if not domains[(objclass, attr)](attr_value):
                    if verbose:
                        print("Attribute value %s not in domain (%s, %s)" % (attr_value, objclass, attr))
                    return False
        return True    

    @classmethod
    @abstractmethod
    def verify_action(cls, action, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def verify_observation(cls, observation, **kwargs):
        pass    


class OOPOMDP_ObjectState(ABC):
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
    

class OOPOMDP_State(ABC):

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
    
