"""Defines the ObservationModel for the 2D Multi-Object Search domain.

Origin: Multi-Object Search using Object-Oriented POMDPs (ICRA 2019)
(extensions: action space changes, different sensor model, gridworld instead of
topological graph)

Observation: {objid : pose(x,y) or NULL}. The sensor model could vary;
             it could be a fan-shaped model as the original paper, or
             it could be something else. But the resulting observation
             should be a map from object id to observed pose or NULL (not observed).

Observation Model

  The agent can observe its own state, as well as object poses
  that are within its sensor range. We only need to model object
  observation.

"""

import pomdp_py
import math
import random
import numpy as np
from pomdp_problems.multi_object_search.domain.state import *
from pomdp_problems.multi_object_search.domain.action import *
from pomdp_problems.multi_object_search.domain.observation import *

#### Observation Models ####
class MosObservationModel(pomdp_py.OOObservationModel):
    """Object-oriented transition model"""
    def __init__(self,
                 dim,
                 sensor,
                 object_ids,
                 sigma=0.01,
                 epsilon=1):
        self.sigma = sigma
        self.epsilon = epsilon
        observation_models = {objid: ObjectObservationModel(objid, sensor, dim,
                                                            sigma=sigma, epsilon=epsilon)
                              for objid in object_ids}
        pomdp_py.OOObservationModel.__init__(self, observation_models)

    def sample(self, next_state, action, argmax=False, **kwargs):
        if not isinstance(action, LookAction):
            return MosOOObservation({})
            # return MosOOObservation({objid: ObjectObservationModel.NULL
            #                          for objid in next_state.object_states
            #                          if objid != next_state.object_states[objid].objclass != "robot"})

        factored_observations = super().sample(next_state, action, argmax=argmax)
        return MosOOObservation.merge(factored_observations, next_state)

class ObjectObservationModel(pomdp_py.ObservationModel):
    def __init__(self, objid, sensor, dim, sigma=0, epsilon=1):
        """
        sigma and epsilon are parameters of the observation model (see paper),
        dim (tuple): a tuple (width, length) for the dimension of the world"""
        self._objid = objid
        self._sensor = sensor
        self._dim = dim
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

        Args:
            observation (ObjectObservation)
            next_state (State)
            action (Action)
        """
        if not isinstance(action, LookAction):
            # No observation should be received
            if observation.pose == ObjectObservation.NULL:
                return 1.0
            else:
                return 0.0

        if observation.objid != self._objid:
            raise ValueError("The observation is not about the same object")

        # The (funny) business of allowing histogram belief update using O(oi|si',sr',a).
        next_robot_state = kwargs.get("next_robot_state", None)
        if next_robot_state is not None:
            assert next_robot_state["id"] == self._sensor.robot_id,\
                "Robot id of observation model mismatch with given state"
            robot_pose = next_robot_state.pose

            if isinstance(next_state, ObjectState):
                assert next_state["id"] == self._objid,\
                    "Object id of observation model mismatch with given state"
                object_pose = next_state.pose
            else:
                object_pose = next_state.pose(self._objid)
        else:
            robot_pose = next_state.pose(self._sensor.robot_id)
            object_pose = next_state.pose(self._objid)

        # Compute the probability
        zi = observation.pose
        alpha, beta, gamma = self._compute_params(self._sensor.within_range(robot_pose, object_pose))

        # Requires Python >= 3.6
        prob = 0.0
        # Event A:
        # object in sensing region and observation comes from object i
        if zi == ObjectObservation.NULL:
            # Even though event A occurred, the observation is NULL.
            # This has 0.0 probability.
            prob += 0.0 * alpha
        else:
            gaussian = pomdp_py.Gaussian(list(object_pose),
                                         [[self.sigma**2, 0],
                                          [0, self.sigma**2]])
            prob += gaussian[zi] * alpha

        # Event B
        prob += (1.0 / self._sensor.sensing_region_size) * beta

        # Event C
        pr_c = 1.0 if zi == ObjectObservation.NULL else 0.0  # indicator zi == NULL
        prob += pr_c * gamma
        return prob


    def sample(self, next_state, action, **kwargs):
        """Returns observation"""
        if not isinstance(action, LookAction):
            # Not a look action. So no observation
            return ObjectObservation(self._objid, ObjectObservation.NULL)

        robot_pose = next_state.pose(self._sensor.robot_id)
        object_pose = next_state.pose(self._objid)

        # Obtain observation according to distribution.
        alpha, beta, gamma = self._compute_params(self._sensor.within_range(robot_pose, object_pose))

        # Requires Python >= 3.6
        event_occured = random.choices(["A", "B", "C"], weights=[alpha, beta, gamma], k=1)[0]
        zi = self._sample_zi(event_occured, next_state)

        return ObjectObservation(self._objid, zi)

    def argmax(self, next_state, action, **kwargs):
        # Obtain observation according to distribution.
        alpha, beta, gamma = self._compute_params(self._sensor.within_range(robot_pose, object_pose))

        event_probs = {"A": alpha,
                       "B": beta,
                       "C": gamma}
        event_occured = max(event_probs, key=lambda e: event_probs[e])
        zi = self._sample_zi(event_occured, next_state, argmax=True)
        return ObjectObservation(self._objid, zi)

    def _sample_zi(self, event, next_state, argmax=False):
        if event == "A":
            object_true_pose = next_state.object_pose(self._objid)
            gaussian =  pomdp_py.Gaussian(list(object_true_pose),
                                          [[self.sigma**2, 0],
                                           [0, self.sigma**2]])
            if not argmax:
                zi = gaussian.random()
            else:
                zi = gaussian.mpe()
            zi = (int(round(zi[0])), int(round(zi[1])))

        elif event == "B":
            # TODO: FIX. zi should ONLY come from the field of view.
            # There is currently no easy way to sample from the field of view.
            width, height = self._dim
            zi = (random.randint(0, width),   # x axis
                  random.randint(0, height))  # y axis
        else: # event == C
            zi = ObjectObservation.NULL
        return zi


### Unit test ###
def unittest():
    from ..env.env import make_laser_sensor,\
        make_proximity_sensor, equip_sensors,\
        interpret, interpret_robot_id
    # Test within search region check,
    # and the observation model probability and
    # sampling functions.
    worldmap =\
        """
        ..........
        ....T.....
        ......x...
        ..T.r.T...
        ..x.......
        ....T.....
        ..........
        """
       #0123456789
       # 10 x 8
    worldstr = equip_sensors(worldmap,
                             {"r": make_laser_sensor(90, (1,5), 0.5, False)})
    env = interpret(worldstr)
    robot_id = interpret_robot_id("r")
    robot_pose = env.state.pose(robot_id)

    # within_range test
    sensor = env.sensors[robot_id]
    assert sensor.within_range(robot_pose, (4,3)) == False
    assert sensor.within_range(robot_pose, (5,3)) == True
    assert sensor.within_range(robot_pose, (6,3)) == True
    assert sensor.within_range(robot_pose, (7,2)) == True
    assert sensor.within_range(robot_pose, (7,3)) == True
    assert sensor.within_range(robot_pose, (4,3)) == False
    assert sensor.within_range(robot_pose, (2,4)) == False
    assert sensor.within_range(robot_pose, (4,1)) == False
    assert sensor.within_range(robot_pose, (4,5)) == False
    assert sensor.within_range(robot_pose, (0,0)) == False

    print(env.state)

    # observation model test
    O0 = ObjectObservationModel(0, sensor, (env.width, env.length), sigma=0.01, epsilon=1)
    O2 = ObjectObservationModel(2, sensor, (env.width, env.length), sigma=0.01, epsilon=1)
    O3 = ObjectObservationModel(3, sensor, (env.width, env.length), sigma=0.01, epsilon=1)
    O5 = ObjectObservationModel(5, sensor, (env.width, env.length), sigma=0.01, epsilon=1)

    z0 = O0.sample(env.state, Look)
    assert z0.pose == ObjectObservation.NULL
    z2 = O2.sample(env.state, Look)
    assert z2.pose == ObjectObservation.NULL
    z3 = O3.sample(env.state, Look)
    assert z3.pose == (6, 3)
    z5 = O5.sample(env.state, Look)
    assert z5.pose == ObjectObservation.NULL

    assert O0.probability(z0, env.state, Look) == 1.0
    assert O2.probability(z2, env.state, Look) == 1.0
    assert O3.probability(z3, env.state, Look) >= 1.0
    assert O3.probability(ObjectObservation(3, ObjectObservation.NULL),
                          env.state, Look) == 0.0
    assert O5.probability(z5, env.state, Look) == 1.0

if __name__ == "__main__":
    unittest()
