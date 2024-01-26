"""
Defines the Observation for the 2D Multi-Object Search domain;

Origin: Multi-Object Search using Object-Oriented POMDPs (ICRA 2019)
(extensions: action space changes, different sensor model, gridworld instead of topological graph)

Observation:

    :code:`{objid : pose(x,y) or NULL}`.
    The sensor model could vary;
    it could be a fan-shaped model as the original paper, or
    it could be something else. But the resulting observation
    should be a map from object id to observed pose or NULL (not observed).
"""
import pomdp_py

###### Observation ######
class ObjectObservation(pomdp_py.Observation):
    """The xy pose of the object is observed; or NULL if not observed"""
    NULL = None
    def __init__(self, objid, pose):
        self.objid = objid
        if type(pose) == tuple and len(pose) == 2\
           or pose == ObjectObservation.NULL:
            self.pose = pose
        else:
            raise ValueError("Invalid observation %s for object"
                             % (str(pose), objid))
    def __hash__(self):
        return hash((self.objid, self.pose))
    def __eq__(self, other):
        if not isinstance(other, ObjectObservation):
            return False
        else:
            return self.objid == other.objid\
                and self.pose == other.pose

class MosOOObservation(pomdp_py.OOObservation):
    """Observation for Mos that can be factored by objects;
    thus this is an OOObservation."""
    def __init__(self, objposes):
        """
        objposes (dict): map from objid to 2d pose or NULL (not ObjectObservation!).
        """
        self._hashcode = hash(frozenset(objposes.items()))
        self.objposes = objposes

    def for_obj(self, objid):
        if objid in self.objposes:
            return ObjectObservation(objid, self.objposes[objid])
        else:
            return ObjectObservation(objid, ObjectObservation.NULL)
        
    def __hash__(self):
        return self._hashcode
    
    def __eq__(self, other):
        if not isinstance(other, MosOOObservation):
            return False
        else:
            return self.objposes == other.objposes

    def __str__(self):
        return "MosOOObservation(%s)" % str(self.objposes)

    def __repr__(self):
        return str(self)

    def factor(self, next_state, *params, **kwargs):
        """Factor this OO-observation by objects"""
        return {objid: ObjectObservation(objid, self.objposes[objid])
                for objid in next_state.object_states
                if objid != next_state.robot_id}
    
    @classmethod
    def merge(cls, object_observations, next_state, *params, **kwargs):
        """Merge `object_observations` into a single OOObservation object;
        
        object_observation (dict): Maps from objid to ObjectObservation"""
        return MosOOObservation({objid: object_observations[objid].pose
                                 for objid in object_observations
                                 if objid != next_state.object_states[objid].objclass != "robot"})
