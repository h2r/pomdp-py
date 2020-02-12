# Defines the Observation/ObservationModel for the 2D Multi-Object Search domain;
#
# Origin: Multi-Object Search using Object-Oriented POMDPs (ICRA 2019)
# (extensions: action space changes,
#              different sensor model,
#              gridworld instead of topological graph)
#
# Observation: {objid : pose(x,y) or NULL}. The sensor model could vary;
#              it could be a fan-shaped model as the original paper, or
#              it could be something else. But the resulting observation
#              should be a map from object id to observed pose or NULL (not observed).
#
# Observation Model
#
#   The agent can observe its own state, as well as object poses
#   that are within its sensor range. We only need to model object
#   observation.

import pomdp_py
import math
import random
import numpy as np
from ..domain.action import *
from ..domain.observation import *

#### Observation Models ####
class MosObservationModel(pomdp_py.OOObservationModel):
    """Object-oriented transition model"""
    def __init__(self,
                 gridworld,
                 sigma=0,
                 epsilon=1):
        self._gridworld = gridworld
        self.sigma = sigma
        self.epsilon = epsilon
        observation_models = {objid: ObjectObservationModel(objid, gridworld,
                                                            sigma=sigma, epsilon=epsilon)
                              for objid in self._gridworld.target_objects}
        pomdp_py.OOObservationModel.__init__(self, observation_models)

    def sample(self, next_state, action, argmax=False, **kwargs):
        if not isinstance(action, LookAction):
            return OOObservation({}, None)
            
        factored_observations = super().sample(next_state, action, argmax=argmax)
        return MosObservation.merge(factored_observations, next_state)

class ObjectObservationModel(pomdp_py.ObservationModel):
    def __init__(self, objid, gridworld, sigma=0, epsilon=1):
        """sigma and epsilon are parameters of the observation model (see paper)"""
        self._objid = objid
        self._gridworld = gridworld  # needed to sample observation
        self.sigma = sigma
        self.epsilon = epsilon

    def _compute_params(self, object_in_sensing_region):
        if object_in_sensing_region:
            # Object is in the sensing region
            alpha = self.epsilon
            beta = (1.0 - self.epsilon) / 2.0
            gamma = (1.0 - self.epsilon) / 2.0
        else:
            # Object is not in the sensing region.
            alpha = (1.0 - self.epsilon) / 2.0
            beta = (1.0 - self.epsilon) / 2.0
            gamma = self.epsilon
        return alpha, beta, gamma
        

    def probability(self, observation, next_state, action, **kwargs):
        """
        Returns the probability of Pr (observation | next_state, action).

        observation (ObjectObservation)
        """
        if observation.objid != self._objid:
            # The observation is not about the same object
            return 0

        zi = observation.pose
        alpha, beta, gamma = self._compute_params(zi == ObjectObservation.NULL)

        # Requires Python >= 3.6
        event_occured = random.choices(["A", "B", "C"], weights=[alpha, beta, gamma], k=1)[0]
        if event_occured == "A":
            # object in sensing region and observation comes from object i
            object_true_pose = next_state.object_pose(self._objid)
            gaussian =  pomdp_py.Gaussian(list(object_true_pose),
                                          np.array([[self.sigma**2, 0],
                                                    [0, self.sigma**2]]))
            return gaussian[zi] * alpha
        elif event_occured == "B":
            return (1.0 / self._gridworld.sensing_region_size) * beta

        else: # event_occured == "C":
            return 1.0 * gamma


    def sample(self, next_state, action, **kwargs):
        """Returns observation"""
        if not isinstance(action, LookAction):
            # Not a look action. So no observation
            return ObjectObservation(self._objid, ObjectObservation.NULL)

        # Obtain observation according to distribution. 
        alpha, beta, gamma = self._compute_params(
            self._gridworld.object_in_sensing_region(next_state.object_pose(self._objid)))

        # Requires Python >= 3.6
        event_occured = random.choices(["A", "B", "C"], weights=[alpha, beta, gamma], k=1)[0]
        zi = self._sample_zi(event_occured, next_state)

        return ObjectObservation(self._objid, zi)

    def argmax(self, next_state, action, **kwargs):
        # Obtain observation according to distribution. 
        alpha, beta, gamma = self._compute_params(
            self._gridworld.object_in_sensing_region(next_state.object_pose(self._objid)))

        event_probs = {"A": alpha,
                       "B": beta,
                       "C": gamma}
        event_occured = max(event_probs, key=lambda e: event_probs[e])
        zi = self._sample_zi(event_occured, next_state, argmax=True)
        return ObjectObservation(self._objid, zi)

    def _sample_zi(self, event, next_state, argmax=False):
        if event_occured == "A":
            object_true_pose = next_state.object_pose(self._objid)
            gaussian =  pomdp_py.Gaussian(list(object_true_pose),
                                          np.array([[self.sigma**2, 0],
                                                    [0, self.sigma**2]]))
            if not argmax:
                zi = self._gridworld.discretize(gaussian.random())
            else:
                zi = self._gridworld.discretize(gaussian.mpe())
                
        elif event_occured == "B":
            zi = (random.randint(0, self._gridworld.width),   # x axis
                  random.randint(0, self._gridworld.height))  # y axis
        else: # event == C
            zi = ObjectObservation.NULL
        return zi
