from pomdp_py.representations.distribution.weighted_particles cimport WeightedParticles
from pomdp_py.framework.basics cimport State, Action, Observation, TransitionModel, ObservationModel

cpdef update_weighted_particles_belief(WeightedParticles current_particles, Action real_action, Observation real_observation, ObservationModel observation_model, TransitionModel transition_model, dict targs=*, dict oargs=*)
                                       
                                       
