# Defines the belief distribution and update for the 2D Multi-Object Search domain;
#
# The belief distribution is represented as a Histogram (or Tabular representation).
# Since the observation only contains mapping from object id to their location,
# the belief update has no leverage on the shape of the sensing region; this is
# makes the belief update algorithm more general to any sensing region but then
# requires updating the belief by iterating over the state space in a nested
# loop. The alternative is to use particle representation but also object-oriented.
# We try both here.
#
# We can directly make use of the Histogram and Particle classes in pomdp_py.
import pomdp_py
import random
from ..domain.state import *

def initialize_belief(dim, robot_ids, object_ids, prior={},
                      representation="histogram", robot_orientations={}, num_particles=100):
    """
    Returns a GenerativeDistribution that is the belief representation for
    the multi-object search problem.

    Args:
        dim (tuple): a tuple (width, length) of the search space gridworld.
        robot_ids (set): a set of robot_ids.
        object_ids (dict): a set of object ids that we want to model the belief distribution
                          over; They are `assumed` to be the target objects, not obstacles,
                          because the robot doesn't really care about obstacle locations and
                          modeling them just adds computation cost.
        prior (dict): A mapping {(objid|robot_id) -> {(x,y) -> [0,1]}}. If used, then 
                      all locations not included in the prior will be treated to have 0 probability.
                      If unspecified for an object, then the belief over that object is assumed
                      to be a uniform distribution.
        robot_orientations (dict): Mapping from robot id to their initial orientation (radian).
                                   Assumed to be 0 if robot id not in this dictionary.
        num_particles (int): Maximum number of particles used to represent the belief

    Returns:
        GenerativeDistribution: the initial belief representation.
    """
    if representation == "histogram":
        return _initialize_histogram_belief(dim, robot_ids, object_ids, prior, robot_orientations)
    elif representation == "particles":
        return _initialize_particles_belief(dim, robot_ids, object_ids,
                                            robot_orientations, num_particles=num_particles)
    else:
        raise ValueError("Unsupported belief representation %s" % representation)

    
def _initialize_histogram_belief(dim, robot_ids, object_ids, prior, robot_orientations):
    """
    Returns the belief distribution represented as a histogram
    """
    all_ids = {**{robot_id : "robot" for robot_id in robot_ids},
               **{objid : "object" for objid in object_ids}}
    oo_hists = {}  # objid -> Histogram
    width, length = dim
    for _id in all_ids:
        hist = {}  # pose -> prob
        total_prob = 0
        if _id in prior:
            # prior knowledge provided. Just use the prior knowledge
            for pose in prior[_id]:
                if all_ids[_id] == "robot":
                    state = RobotState(_id, pose, (), None)
                else:  # object
                    state = ObjectState(_id, "target", pose)
                hist[state] = prior[_id][pose]
                total_prob += hist[state]
        else:
            # no prior knowledge. So uniform.
            for x in range(width):
                for y in range(length):
                    if all_ids[_id] == "robot":
                        if _id in robot_orientations:
                            pose = (x,y,robot_orientations[_id])
                        else:
                            pose = (x,y,0)
                        state = RobotState(_id, pose, (), None)
                    else:  # object
                        pose = (x,y)
                        state = ObjectState(_id, "target", pose)
                    hist[state] = 1.0
                    total_prob += hist[state]

        # Normalize
        for state in hist:
            hist[state] /= total_prob

        hist_belief = pomdp_py.Histogram(hist)
        oo_hists[_id] = hist_belief
    return pomdp_py.OOBelief(oo_hists)


def _initialize_particles_belief(dim, robot_ids, object_ids, prior, robot_orientations, num_particles=100):
    all_ids = {**{robot_id : "robot" for robot_id in robot_ids},
               **{objid : "object" for objid in object_ids}}
    oo_particles = {}  # objid -> Particles
    width, length = dim
    for _id in all_ids:
        particles = []  # list of states
        if _id in prior:
            # prior knowledge provided. Just use the prior knowledge
            prior_sum = sum(prior[_id][pose] for pose in prior[_id])
            for pose in prior[_id]:
                if all_ids[_id] == "robot":
                    state = RobotState(_id, pose, (), None)
                else:  # object
                    state = ObjectState(_id, "target", pose)
                amount_to_add = (prior[_id][pose] / prior_sum) * num_particles
                for _ in range(amount_to_add):
                    particles.append(state)
        else:
            # no prior knowledge. So uniformly sample `num_particles` number of states.
            for _ in range(num_particles):
                x = random.randrange(0, width)
                y = random.randrange(0, length)
                if all_ids[_id] == "robot":
                    if _id in robot_orientations:
                        pose = (x,y,robot_orientations[_id])
                    else:
                        pose = (x,y,0)
                    state = RobotState(_id, pose, (), None)
                else:  # object
                    pose = (x,y)
                    state = ObjectState(_id, "target", pose)
                particles.append(state)

        particles_belief = pomdp_py.Particles(particles)
        oo_particles[_id] = particles_belief
    return pomdp_py.OOBelief(oo_particles)
