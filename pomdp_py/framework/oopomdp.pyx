"""This module describes components of the
OO-POMDP interface in `pomdp_py`.

An OO-POMDP is a specific type of POMDP where
the state and observation spaces are factored
by objects. As a result, the transition, observation,
and belief distributions are all factored by objects.
A main benefit of using OO-POMDP is that the
object factoring reduces the scaling of belief
space from exponential to linear as the number
of objects increases. See :cite:`wandzel2019multi`."""

from pomdp_py.framework.basics cimport POMDP, State, Action, Observation,\
    ObservationModel, TransitionModel, GenerativeDistribution
from pomdp_py.utils.cython_utils cimport det_dict_hash
import collections
import copy

cdef class OOPOMDP(POMDP):
    """
    An OO-POMDP is again defined by an agent and an environment.

    __init__(self, agent, env, name="OOPOMDP")
    """
    def __init__(self, agent, env, name="OOPOMDP"):
        super().__init__(agent, env, name=name)

cdef class ObjectState(State):
    """
    This is the result of OOState factoring; A state
    in an OO-POMDP is made up of ObjectState(s), each with
    an `object class` (str) and a set of `attributes` (dict).
    """
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
        self._hashcache = -1

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '%s::(%s,%s)' % (str(self.__class__.__name__),
                                str(self.objclass),
                                str(self.attributes))

    def __hash__(self):
        """__hash__(self)
        Hash the ObjectState.
        For more efficient hashing for your specific domain, please overwrite this function."""
        if self._hashcache == -1:
            self._hashcache = det_dict_hash(self.attributes)
        return self._hashcache

    def __eq__(self, other):
        if not isinstance(other, ObjectState):
            return False
        return self.objclass == other.objclass\
            and self.attributes == other.attributes

    def __getitem__(self, attr):
        """__getitem__(self, attr)
        Returns the attribute"""
        return self.attributes[attr]

    def __setitem__(self, attr, value):
        """__setitem__(self, attr, value)
        Sets the attribute `attr` to the given value."""
        self.attributes[attr] = value

    def __len__(self):
        return len(self.attributes)

    def copy(self):
        """copy(self)
        Copies this ObjectState."""
        return ObjectState(self.objclass, copy.deepcopy(self.attributes))


cdef class OOState(State):

    """
    State that can be factored by objects, that is, to ObjectState(s).

    Note: to change the state of an object, you can use set_object_state.  Do
    not directly assign the object state by e.g. oostate.object_states[objid] =
    object_state, because it will cause the hashcode to be incorrect in the
    oostate after the change.

    __init__(self, object_states)
    """

    def __init__(self, object_states):
        """
        objects_states: dictionary of dictionaries; Each dictionary represents an object state:
            { ID: ObjectState }
        """
        # internally, objects are sorted by ID.
        self.object_states = object_states
        self._hashcache = -1

    @property
    def situation(self):
        """situation(selppf)
        This is a `frozenset` which can be used to identify
        the situation of this state since it supports hashing."""
        return frozenset(self.object_states.items())

    def __str__(self):
        return '%s::[%s]' % (str(self.__class__.__name__),
                             str(self.object_states))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, OOState)\
            and self.object_states == other.object_states

    def __hash__(self):
        """__hash__(self)
        Hash the ObjectState.
        For more efficient hashing for your specific domain, please overwrite this function."""
        if self._hashcache == -1:
            self._hashcache = det_dict_hash(self.object_states)
        return self._hashcache

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_object_state(self, objid):
        """get_object_state(self, objid)
        Returns the ObjectState for given object."""
        return self.object_states[objid]

    def set_object_state(self, objid, object_state):
        """set_object_state(self, objid, object_state)
        Sets the state of the given
        object to be the given object state (ObjectState)
        """
        # Deprecation Warning: set_object_state is not valid because OOState is not mutable.
        self.object_states[objid] = object_state
        self._hashcache = -1

    def get_object_class(self, objid):
        """get_object_class(self, objid)
        Returns the class of requested object"""
        return self.object_states[objid].objclass

    def get_object_attribute(self, objid, attr):
        """get_object_attribute(self, objid, attr)
        Returns the attributes of requested object"""
        return self.object_states[objid][attr]

    def copy(self):
        """copy(self)
        Copies the state."""
        return OOState(copy.deepcopy(self.object_states))

    def s(self, objid):
        """convenient alias function"""
        return self.object_states[objid]

cdef class OOTransitionModel(TransitionModel):

    """
    :math:`T(s' | s, a) = \prod_i T(s_i' | s, a)`

    __init__(self, transition_models):
    Args:
        transition_models (dict) objid -> transition_model
    """

    def __init__(self, transition_models):
        """
        transition_models (dict) objid -> transition_model
        """
        self._transition_models = transition_models

    def probability(self, next_state, state, action, **kwargs):
        """probability(self, next_state, state, action, **kwargs)
        Returns :math:`T(s' | s, a)
        """
        if not isinstance(next_state, OOState):
            raise ValueError("next_state must be OOState")
        if not isinstance(state, OOState):
            raise ValueError("state must be OOState")
        if next_state.object_states.keys() != state.object_states.keys():
            raise ValueError("object types modified between states")
        trans_prob = 1.0
        for objid in self._transition_models:
            object_state = state.object_states[objid]
            next_object_state = next_state.object_states[objid]
            trans_prob = trans_prob * self._transition_models[objid].probability(next_object_state, state, action, **kwargs)
        return trans_prob

    def sample(self, state, action, argmax=False, **kwargs):
        """
        sample(self, state, action, argmax=False, **kwargs)
        Returns random next_state"""
        if not isinstance(state, OOState):
            raise ValueError("state must be OOState")
        object_states = {}
        for objid in state.object_states:
            if objid not in self._transition_models:
                # no transition model provided for this object. Then no transition happens.
                object_states[objid] = copy.deepcopy(state.object_states[objid])
                continue
            if argmax:
                next_object_state = self._transition_models[objid].argmax(state, action, **kwargs)
            else:
                next_object_state = self._transition_models[objid].sample(state, action, **kwargs)
            object_states[objid] = next_object_state
        return OOState(object_states)

    def argmax(self, state, action, **kwargs):
        """
        argmax(self, state, action, **kwargs)
        Returns the most likely next state"""
        return self.sample(state, action, argmax=True, **kwargs)

    def __getitem__(self, objid):
        """__getitem__(self, objid)
        Returns transition model for given object"""
        return self._transition_models[objid]

    @property
    def transition_models(self):
        """transition_models(self)"""
        return self._transition_models

cdef class OOObservation(Observation):
    def factor(self, next_state, action, **kwargs):
        """factor(self, next_state, action, **kwargs)
        Factors the observation by objects.
        That is, :math:`z\mapsto z_1,\cdots,z_n`

        Args:
            next_state (OOState): given state
            action (Action): given action
        Returns:
            dict: map from object id to a `pomdp_py.Observation`.
        """
        raise NotImplemented
    @classmethod
    def merge(cls, object_observations, next_state, action, **kwargs):
        """merge(cls, object_observations, next_state, action, **kwargs)
        Merges the factored `object_observations` into a
        single OOObservation.

        Args:
            object_observations (dict): map from object id to a `pomdp_py.Observation`.
            next_state (OOState): given state
            action (Action): given action
        Returns:
            OOObservation: the merged observation.
        """
        raise NotImplemented

cdef class OOObservationModel(ObservationModel):

    """
    :math:`O(z | s', a) = \prod_i O(z_i' | s', a)`

    __init__(self, observation_models):
    Args:
        observation_models (dict) objid -> observation_model
    """

    def __init__(self, observation_models, merge_func=None):#, factor_observation_func, merge_observations_func):
        """
        observation_models (dict) objid -> observation_model
        factor_observation_func: (observation, objid, next_state, action) -> {objid->observations}
        merge_observations_func: (factored_observations, next_state, action) -> observation
        """
        self._observation_models = observation_models
        self._merge_func = merge_func


    def probability(self, observation, next_state, action, **kwargs):
        """
        probability(self, observation, next_state, action, **kwargs)
        Returns :math:`O(z | s', a)`.
        """
        if not isinstance(next_state, OOState):
            raise ValueError("state must be OOState")
        obsrv_prob = 1.0
        factored_observations = observation.factor(next_state, action) #self._factor_observation_func(observation, next_state)
        for objid in next_state.object_states:
            obsrv_prob = obsrv_prob * self._observation_models[objid].probability(factored_observations[objid],
                                                                                  next_state, action, **kwargs)
        return obsrv_prob

    def sample(self, next_state, action, argmax=False, **kwargs):
        """
        sample(self, next_state, action, argmax=False, **kwargs)
        Returns random observation"""
        if not isinstance(next_state, OOState):
            raise ValueError("state must be OOState")
        factored_observations = {}
        for objid in self._observation_models:
            if not argmax:
                observation = self._observation_models[objid].sample(next_state,
                                                                     action, **kwargs)
            else:
                observation = self._observation_models[objid].argmax(next_state,
                                                                     action, **kwargs)
            factored_observations[objid] = observation
        if self._merge_func is None:
            return factored_observations
        else:
            return self._merge_func(factored_observations, next_state, action, **kwargs)

    def argmax(self, next_state, action, **kwargs):
        """
        argmax(self, next_state, action, **kwargs)
        Returns most likely observation"""
        return self.sample(next_state, action, argmax=True, **kwargs)

    def __getitem__(self, objid):
        """__getitem__(self, objid)
        Returns observation model for given object"""
        return self._observation_models[objid]

    @property
    def observation_models(self):
        """observation_models(self)"""
        return self._observation_models


cdef class OOBelief(GenerativeDistribution):
    """
    Belief factored by objects.
    """
    def __init__(self, object_beliefs):
        """
        object_beliefs (objid -> GenerativeDistribution)
        """
        self._object_beliefs = object_beliefs

    def __getitem__(self, state):
        """__getitem__(self, state)
        Returns belief probability of given state"""
        if not isinstance(state, OOState):
            raise ValueError("state must be OOState")
        belief_prob = 1.0
        for objid in self._object_beliefs:
            object_state = state.object_states[objid]
            belief_prob = belief_prob * self._object_beliefs[objid].probability(object_state)
        return belief_prob

    def random(self, return_oostate=True, **kwargs):
        """random(self, return_oostate=False, **kwargs)
        Returns a random state"""
        object_states = {}
        for objid in self._object_beliefs:
            object_states[objid] = self._object_beliefs[objid].random(**kwargs)
        if return_oostate:
            return OOState(object_states)
        else:
            return object_states

    def mpe(self, return_oostate=True, **kwargs):
        """mpe(self, return_oostate=False, **kwargs)
        Returns most likely state."""
        object_states = {}
        for objid in self._object_beliefs:
            object_states[objid] = self._object_beliefs[objid].mpe(**kwargs)
        if return_oostate:
            return OOState(object_states)
        else:
            return object_states

    def __setitem__(self, oostate, value):
        """__setitem__(self, oostate, value)
        Sets the probability of a given `oostate` to `value`.
        Note always feasible."""
        raise NotImplemented

    def object_belief(self, objid):
        """object_belief(self, objid)
        Returns the belief (GenerativeDistribution) for the given object."""
        return self._object_beliefs[objid]

    def set_object_belief(self, objid, belief):
        """set_object_belief(self, objid, belief)
        Sets the belief of object to be the given `belief` (GenerativeDistribution)"""
        self._object_beliefs[objid] = belief

    @property
    def object_beliefs(self):
        """object_beliefs(self)"""
        return self._object_beliefs

    def b(self, objid):
        """convenient alias function call"""
        return self._object_beliefs[objid]
